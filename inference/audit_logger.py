"""
Audit logging for all inference requests.
Stores request/response data with tenant isolation for compliance.
"""
import json
import uuid
import sqlite3
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from threading import Lock

from loguru import logger


class AuditLogger:
    """SQLite-based audit logger for inference requests."""

    def __init__(self, db_path: str = "logs/audit.db"):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_db()

    def _init_db(self):
        """Initialize the audit database."""
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                user_message TEXT,
                model_response TEXT,
                model_type TEXT,
                model_version TEXT,
                adapter_key TEXT,
                is_canary INTEGER DEFAULT 0,
                use_rag INTEGER DEFAULT 1,
                retrieval_method TEXT,
                citations_count INTEGER DEFAULT 0,
                grounding_score REAL,
                latency_ms REAL,
                retrieval_time_ms REAL,
                generation_time_ms REAL,
                token_count INTEGER,
                was_refused INTEGER DEFAULT 0,
                was_blocked INTEGER DEFAULT 0,
                error TEXT,
                metadata TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                request_id TEXT,
                tenant_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                rating INTEGER,
                feedback_type TEXT,
                comment TEXT,
                flagged_issues TEXT,
                FOREIGN KEY (request_id) REFERENCES requests(request_id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_tenant ON requests(tenant_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_tenant ON feedback(tenant_id)")
        conn.commit()
        conn.close()

    def log_request(
        self,
        tenant_id: str,
        user_message: str,
        model_response: str,
        model_type: str = "",
        model_version: str = "",
        adapter_key: str = "",
        is_canary: bool = False,
        use_rag: bool = True,
        retrieval_method: str = "",
        citations_count: int = 0,
        grounding_score: float = None,
        latency_ms: float = 0,
        retrieval_time_ms: float = 0,
        generation_time_ms: float = 0,
        token_count: int = 0,
        was_refused: bool = False,
        was_blocked: bool = False,
        error: str = None,
        metadata: Dict = None,
    ) -> str:
        """Log an inference request. Returns request_id."""
        request_id = str(uuid.uuid4())[:12]

        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                """INSERT INTO requests VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )""",
                (
                    request_id,
                    tenant_id,
                    datetime.utcnow().isoformat(),
                    (user_message or "")[:2000],
                    (model_response or "")[:5000],
                    model_type,
                    model_version,
                    adapter_key,
                    int(is_canary),
                    int(use_rag),
                    retrieval_method,
                    citations_count,
                    grounding_score,
                    latency_ms,
                    retrieval_time_ms,
                    generation_time_ms,
                    token_count,
                    int(was_refused),
                    int(was_blocked),
                    error,
                    json.dumps(metadata or {}),
                ),
            )
            conn.commit()
            conn.close()

        return request_id

    def log_feedback(
        self,
        request_id: str,
        tenant_id: str,
        rating: int,
        feedback_type: str = "thumbs",
        comment: str = None,
        flagged_issues: List[str] = None,
    ) -> str:
        """Log user feedback. Returns feedback_id."""
        feedback_id = str(uuid.uuid4())[:12]

        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    feedback_id,
                    request_id,
                    tenant_id,
                    datetime.utcnow().isoformat(),
                    rating,
                    feedback_type,
                    comment,
                    json.dumps(flagged_issues or []),
                ),
            )
            conn.commit()
            conn.close()

        return feedback_id

    def get_request_stats(self, tenant_id: str = None, hours: int = 24) -> Dict:
        """Get request statistics."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row

        if tenant_id:
            rows = conn.execute(
                """SELECT
                    COUNT(*) as total,
                    AVG(latency_ms) as avg_latency,
                    MAX(latency_ms) as max_latency,
                    MIN(latency_ms) as min_latency,
                    SUM(was_refused) as refusals,
                    SUM(was_blocked) as blocked,
                    SUM(is_canary) as canary_requests,
                    AVG(grounding_score) as avg_grounding
                FROM requests WHERE tenant_id = ?
                AND timestamp >= datetime('now', ?)""",
                (tenant_id, f"-{hours} hours"),
            ).fetchone()
        else:
            rows = conn.execute(
                """SELECT
                    COUNT(*) as total,
                    AVG(latency_ms) as avg_latency,
                    MAX(latency_ms) as max_latency,
                    MIN(latency_ms) as min_latency,
                    SUM(was_refused) as refusals,
                    SUM(was_blocked) as blocked,
                    SUM(is_canary) as canary_requests,
                    AVG(grounding_score) as avg_grounding
                FROM requests
                WHERE timestamp >= datetime('now', ?)""",
                (f"-{hours} hours",),
            ).fetchone()

        if tenant_id:
            fb_rows = conn.execute(
                """SELECT AVG(rating) as avg_rating, COUNT(*) as feedback_count
                FROM feedback WHERE tenant_id = ?""",
                (tenant_id,),
            ).fetchone()
        else:
            fb_rows = conn.execute(
                "SELECT AVG(rating) as avg_rating, COUNT(*) as feedback_count FROM feedback"
            ).fetchone()

        conn.close()

        return {
            "total_requests": rows["total"] if rows else 0,
            "avg_latency_ms": round(rows["avg_latency"] or 0, 1) if rows else 0,
            "max_latency_ms": round(rows["max_latency"] or 0, 1) if rows else 0,
            "min_latency_ms": round(rows["min_latency"] or 0, 1) if rows else 0,
            "refusals": rows["refusals"] or 0 if rows else 0,
            "blocked": rows["blocked"] or 0 if rows else 0,
            "canary_requests": rows["canary_requests"] or 0 if rows else 0,
            "avg_grounding_score": round(rows["avg_grounding"] or 0, 3) if rows else 0,
            "avg_feedback_rating": round(fb_rows["avg_rating"] or 0, 2) if fb_rows else 0,
            "feedback_count": fb_rows["feedback_count"] or 0 if fb_rows else 0,
            "tenant_id": tenant_id or "all",
            "period_hours": hours,
        }

    def get_recent_requests(self, tenant_id: str = None, limit: int = 20) -> List[Dict]:
        """Get recent requests for monitoring."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row

        if tenant_id:
            rows = conn.execute(
                """SELECT request_id, tenant_id, timestamp, user_message,
                   model_type, latency_ms, was_refused, grounding_score
                FROM requests WHERE tenant_id = ?
                ORDER BY timestamp DESC LIMIT ?""",
                (tenant_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT request_id, tenant_id, timestamp, user_message,
                   model_type, latency_ms, was_refused, grounding_score
                FROM requests ORDER BY timestamp DESC LIMIT ?""",
                (limit,),
            ).fetchall()

        conn.close()
        return [dict(row) for row in rows]


# Global singleton
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
