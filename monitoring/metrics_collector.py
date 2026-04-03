"""
Metrics collection from all system components.
Aggregates data from audit logs, model registry, and system resources.
"""
import json
import sqlite3
import psutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from loguru import logger


@dataclass
class SystemMetrics:
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_used_percent: float
    gpu_allocated_gb: Optional[float] = None
    gpu_total_gb: Optional[float] = None


@dataclass
class TenantMetrics:
    tenant_id: str
    period_hours: int
    total_requests: int
    avg_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float
    error_count: int
    refusal_count: int
    canary_count: int
    avg_grounding_score: float
    avg_feedback_rating: float
    feedback_count: int
    positive_feedback: int
    negative_feedback: int


@dataclass
class ModelHealth:
    tenant_id: str
    model_type: str
    adapter_path: str
    status: str
    last_train_loss: Optional[float]
    last_eval_loss: Optional[float]
    version: str
    registered_at: str


class MetricsCollector:
    """Collects and aggregates metrics from all sources."""

    def __init__(self, audit_db_path: str = "logs/audit.db"):
        self.audit_db_path = audit_db_path

    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system resource metrics."""
        memory = psutil.virtual_memory()

        gpu_allocated = None
        gpu_total = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu_allocated = round(torch.cuda.memory_allocated() / 1024**3, 2)
                gpu_total = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        except ImportError:
            pass

        disk = psutil.disk_usage('/')

        return SystemMetrics(
            timestamp=datetime.utcnow().isoformat(),
            cpu_percent=psutil.cpu_percent(interval=0.5),
            memory_percent=memory.percent,
            memory_used_gb=round(memory.used / 1024**3, 2),
            memory_available_gb=round(memory.available / 1024**3, 2),
            disk_used_percent=disk.percent,
            gpu_allocated_gb=gpu_allocated,
            gpu_total_gb=gpu_total,
        )

    def collect_tenant_metrics(
        self,
        tenant_id: str,
        hours: int = 24,
    ) -> TenantMetrics:
        """Collect request metrics for a specific tenant."""
        empty = TenantMetrics(
            tenant_id=tenant_id, period_hours=hours,
            total_requests=0, avg_latency_ms=0, p95_latency_ms=0,
            max_latency_ms=0, error_count=0, refusal_count=0,
            canary_count=0, avg_grounding_score=0,
            avg_feedback_rating=0, feedback_count=0,
            positive_feedback=0, negative_feedback=0,
        )

        if not Path(self.audit_db_path).exists():
            return empty

        try:
            conn = sqlite3.connect(self.audit_db_path)
            conn.row_factory = sqlite3.Row

            cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

            row = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    COALESCE(AVG(latency_ms), 0) as avg_lat,
                    COALESCE(MAX(latency_ms), 0) as max_lat,
                    COALESCE(SUM(CASE WHEN error IS NOT NULL AND error != '' THEN 1 ELSE 0 END), 0) as errors,
                    COALESCE(SUM(was_refused), 0) as refusals,
                    COALESCE(SUM(is_canary), 0) as canary,
                    COALESCE(AVG(grounding_score), 0) as avg_ground
                FROM requests
                WHERE tenant_id = ? AND timestamp >= ?
            """, (tenant_id, cutoff)).fetchone()

            # P95 latency
            p95_rows = conn.execute("""
                SELECT latency_ms FROM requests
                WHERE tenant_id = ? AND timestamp >= ? AND latency_ms IS NOT NULL
                ORDER BY latency_ms ASC
            """, (tenant_id, cutoff)).fetchall()

            p95_latency = 0.0
            if p95_rows:
                idx = min(int(len(p95_rows) * 0.95), len(p95_rows) - 1)
                p95_latency = p95_rows[idx]["latency_ms"]

            # Feedback metrics
            fb_row = conn.execute("""
                SELECT
                    COALESCE(AVG(rating), 0) as avg_rating,
                    COUNT(*) as fb_count,
                    COALESCE(SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END), 0) as positive,
                    COALESCE(SUM(CASE WHEN rating <= 2 THEN 1 ELSE 0 END), 0) as negative
                FROM feedback
                WHERE tenant_id = ? AND timestamp >= ?
            """, (tenant_id, cutoff)).fetchone()

            conn.close()

            return TenantMetrics(
                tenant_id=tenant_id,
                period_hours=hours,
                total_requests=row["total"],
                avg_latency_ms=round(row["avg_lat"], 1),
                p95_latency_ms=round(p95_latency, 1),
                max_latency_ms=round(row["max_lat"], 1),
                error_count=row["errors"],
                refusal_count=row["refusals"],
                canary_count=row["canary"],
                avg_grounding_score=round(row["avg_ground"], 3),
                avg_feedback_rating=round(fb_row["avg_rating"], 2),
                feedback_count=fb_row["fb_count"],
                positive_feedback=fb_row["positive"],
                negative_feedback=fb_row["negative"],
            )
        except Exception as e:
            logger.warning(f"Failed to collect metrics for {tenant_id}: {e}")
            return empty

    def collect_model_health(self) -> List[ModelHealth]:
        """Collect model health from registry."""
        registry_path = Path("models/registry.json")
        if not registry_path.exists():
            return []

        with open(registry_path) as f:
            registry = json.load(f)

        models = []
        for m in registry.get("models", []):
            metrics = m.get("metrics", {})
            models.append(ModelHealth(
                tenant_id=m.get("tenant_id", ""),
                model_type=m.get("model_type", ""),
                adapter_path=m.get("adapter_path", ""),
                status=m.get("status", "unknown"),
                last_train_loss=metrics.get("train_loss"),
                last_eval_loss=metrics.get("eval_loss") or metrics.get("dpo_eval_loss"),
                version=m.get("version", ""),
                registered_at=m.get("registered_at", ""),
            ))

        return models

    def collect_latency_timeseries(
        self,
        tenant_id: str = None,
        hours: int = 24,
        bucket_minutes: int = 30,
    ) -> List[Dict]:
        """Collect latency timeseries data for charts."""
        if not Path(self.audit_db_path).exists():
            return []

        try:
            conn = sqlite3.connect(self.audit_db_path)
            cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

            query = """
                SELECT
                    strftime('%Y-%m-%dT%H:', timestamp) ||
                        CAST((CAST(strftime('%M', timestamp) AS INTEGER) / ?) * ? AS TEXT) as bucket,
                    COUNT(*) as count,
                    AVG(latency_ms) as avg_lat,
                    MAX(latency_ms) as max_lat,
                    AVG(grounding_score) as avg_ground
                FROM requests
                WHERE timestamp >= ?
            """
            params = [bucket_minutes, bucket_minutes, cutoff]

            if tenant_id:
                query += " AND tenant_id = ?"
                params.append(tenant_id)

            query += " GROUP BY bucket ORDER BY bucket"

            rows = conn.execute(query, params).fetchall()
            conn.close()

            return [
                {
                    "timestamp": row[0],
                    "request_count": row[1],
                    "avg_latency_ms": round(row[2] or 0, 1),
                    "max_latency_ms": round(row[3] or 0, 1),
                    "avg_grounding": round(row[4] or 0, 3),
                }
                for row in rows
            ]
        except Exception as e:
            logger.warning(f"Failed to collect timeseries: {e}")
            return []

    def collect_all(self, hours: int = 24) -> Dict:
        """Collect all metrics in one call."""
        system = self.collect_system_metrics()
        tenants = {}
        for tid in ["sis", "mfg"]:
            tenants[tid] = asdict(self.collect_tenant_metrics(tid, hours))

        models = [asdict(m) for m in self.collect_model_health()]
        timeseries = self.collect_latency_timeseries(hours=hours)

        return {
            "collected_at": datetime.utcnow().isoformat(),
            "period_hours": hours,
            "system": asdict(system),
            "tenants": tenants,
            "models": models,
            "timeseries": timeseries,
        }


# Singleton
_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector
