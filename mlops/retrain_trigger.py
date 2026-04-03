"""
Automated retraining trigger.
Monitors metrics and flags tenants that need retraining.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from loguru import logger

from monitoring.metrics_collector import get_metrics_collector


@dataclass
class RetrainRecommendation:
    tenant_id: str
    priority: str  # "high", "medium", "low"
    reasons: List[str]
    recommended_action: str
    metrics_snapshot: Dict
    timestamp: str


@dataclass
class RetrainThresholds:
    min_negative_feedback: int = 5
    max_avg_feedback: float = 3.0
    min_grounding_score: float = 0.5
    min_new_documents: int = 5
    max_eval_loss_increase: float = 0.1
    check_period_hours: int = 24


THRESHOLDS = RetrainThresholds()


def check_retrain_needed(tenant_id: str, hours: int = None) -> Optional[RetrainRecommendation]:
    """
    Check if a tenant's model needs retraining.

    Triggers:
    1. High negative feedback rate
    2. Low average feedback rating
    3. Declining grounding scores
    4. New documents added to tenant data
    5. Evaluation regression detected
    """
    hours = hours or THRESHOLDS.check_period_hours
    collector = get_metrics_collector()
    metrics = collector.collect_tenant_metrics(tenant_id, hours)

    reasons = []
    priority = "low"

    # Check 1: Negative feedback
    if metrics.negative_feedback >= THRESHOLDS.min_negative_feedback:
        reasons.append(
            f"High negative feedback: {metrics.negative_feedback} thumbs-down "
            f"(threshold: {THRESHOLDS.min_negative_feedback})"
        )
        priority = "medium"

    # Check 2: Low average rating
    if metrics.feedback_count >= 3 and metrics.avg_feedback_rating < THRESHOLDS.max_avg_feedback:
        reasons.append(
            f"Low feedback rating: {metrics.avg_feedback_rating:.1f} "
            f"(threshold: {THRESHOLDS.max_avg_feedback})"
        )
        priority = "high"

    # Check 3: Low grounding
    if metrics.total_requests >= 5 and metrics.avg_grounding_score < THRESHOLDS.min_grounding_score:
        reasons.append(
            f"Low grounding score: {metrics.avg_grounding_score:.2f} "
            f"(threshold: {THRESHOLDS.min_grounding_score})"
        )
        priority = "high"

    # Check 4: New documents
    new_docs = _check_new_documents(tenant_id)
    if new_docs >= THRESHOLDS.min_new_documents:
        reasons.append(
            f"New documents detected: {new_docs} "
            f"(threshold: {THRESHOLDS.min_new_documents})"
        )
        if priority == "low":
            priority = "medium"

    # Check 5: Regression in eval
    regression = _check_eval_regression(tenant_id)
    if regression:
        reasons.append(f"Evaluation regression detected: {regression}")
        priority = "high"

    if not reasons:
        return None

    if priority == "high":
        action = "Immediate retraining recommended. Rebuild SFT dataset, retrain adapter, run evaluation."
    elif priority == "medium":
        action = "Retraining recommended at next maintenance window. Review flagged feedback for data curation."
    else:
        action = "Monitor. Consider retraining when more data is available."

    recommendation = RetrainRecommendation(
        tenant_id=tenant_id,
        priority=priority,
        reasons=reasons,
        recommended_action=action,
        metrics_snapshot={
            "total_requests": metrics.total_requests,
            "avg_feedback_rating": metrics.avg_feedback_rating,
            "negative_feedback": metrics.negative_feedback,
            "avg_grounding_score": metrics.avg_grounding_score,
            "error_count": metrics.error_count,
        },
        timestamp=datetime.utcnow().isoformat(),
    )

    logger.info(
        f"Retrain recommendation for {tenant_id}: priority={priority}, "
        f"reasons={len(reasons)}"
    )

    return recommendation


def _check_new_documents(tenant_id: str) -> int:
    """Check if new documents have been added since last training."""
    chunks_path = Path(f"data/{tenant_id}/chunks/chunks.json")
    metadata_path = Path(f"models/adapters/{tenant_id}/sft/training_metadata.json")

    if not chunks_path.exists() or not metadata_path.exists():
        return 0

    chunks_mtime = chunks_path.stat().st_mtime
    metadata_mtime = metadata_path.stat().st_mtime

    if chunks_mtime > metadata_mtime:
        with open(chunks_path) as f:
            chunks = json.load(f)
        return len(chunks)

    return 0


def _check_eval_regression(tenant_id: str) -> Optional[str]:
    """Check for evaluation score regression."""
    reports_dir = Path("evaluation/reports")
    if not reports_dir.exists():
        return None

    reports = sorted(
        reports_dir.glob(f"golden_set_{tenant_id}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if len(reports) < 2:
        return None

    with open(reports[0]) as f:
        latest = json.load(f)
    with open(reports[1]) as f:
        previous = json.load(f)

    latest_rate = latest.get("pass_rate", 0)
    previous_rate = previous.get("pass_rate", 0)

    if previous_rate - latest_rate > 0.1:
        return f"Pass rate dropped from {previous_rate:.0%} to {latest_rate:.0%}"

    return None


def check_all_tenants() -> Dict[str, Optional[RetrainRecommendation]]:
    """Check all tenants for retraining needs."""
    results = {}
    for tenant_id in ["sis", "mfg"]:
        recommendation = check_retrain_needed(tenant_id)
        results[tenant_id] = recommendation

        if recommendation:
            priority_icon = {"high": "[HIGH]", "medium": "[MED]", "low": "[LOW]"}
            logger.info(
                f"{priority_icon.get(recommendation.priority, '[ ? ]')} "
                f"[{tenant_id}] {recommendation.priority.upper()}: "
                f"{', '.join(recommendation.reasons)}"
            )

    # Save results
    report_dir = Path("evaluation/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checked_at": datetime.utcnow().isoformat(),
        "recommendations": {
            tid: asdict(r) if r else None
            for tid, r in results.items()
        },
    }
    report_path = report_dir / f"retrain_check_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Retrain check saved: {report_path}")

    return results


if __name__ == "__main__":
    results = check_all_tenants()
    for tid, rec in results.items():
        if rec:
            print(f"\n{'='*50}")
            print(f"Tenant: {tid.upper()}")
            print(f"Priority: {rec.priority}")
            print(f"Reasons: {rec.reasons}")
            print(f"Action: {rec.recommended_action}")
        else:
            print(f"\n[{tid}] No retraining needed")
