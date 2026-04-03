"""
Alerting rules and triggers.
Checks metrics against thresholds and fires alerts.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from loguru import logger

from monitoring.metrics_collector import get_metrics_collector


@dataclass
class Alert:
    alert_id: str
    severity: str  # "critical", "warning", "info"
    category: str
    tenant_id: str
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    triggered_at: str
    resolved: bool = False


@dataclass
class AlertRule:
    name: str
    metric: str
    operator: str  # "gt", "lt", "eq"
    threshold: float
    severity: str
    category: str
    message_template: str


# Alert rules
ALERT_RULES: List[AlertRule] = [
    AlertRule(
        name="high_latency",
        metric="avg_latency_ms",
        operator="gt",
        threshold=5000,
        severity="warning",
        category="performance",
        message_template="Average latency {value:.0f}ms exceeds threshold {threshold:.0f}ms for tenant {tenant_id}",
    ),
    AlertRule(
        name="critical_latency",
        metric="p95_latency_ms",
        operator="gt",
        threshold=10000,
        severity="critical",
        category="performance",
        message_template="P95 latency {value:.0f}ms exceeds threshold {threshold:.0f}ms for tenant {tenant_id}",
    ),
    AlertRule(
        name="low_grounding",
        metric="avg_grounding_score",
        operator="lt",
        threshold=0.5,
        severity="warning",
        category="quality",
        message_template="Average grounding score {value:.2f} below threshold {threshold:.2f} for tenant {tenant_id}",
    ),
    AlertRule(
        name="high_error_rate",
        metric="error_count",
        operator="gt",
        threshold=10,
        severity="critical",
        category="reliability",
        message_template="{value:.0f} errors detected (threshold: {threshold:.0f}) for tenant {tenant_id}",
    ),
    AlertRule(
        name="low_feedback",
        metric="avg_feedback_rating",
        operator="lt",
        threshold=2.5,
        severity="warning",
        category="quality",
        message_template="Average feedback rating {value:.1f} below threshold {threshold:.1f} for tenant {tenant_id}",
    ),
    AlertRule(
        name="high_refusal_rate",
        metric="refusal_count",
        operator="gt",
        threshold=20,
        severity="info",
        category="safety",
        message_template="{value:.0f} refusals detected (threshold: {threshold:.0f}) for tenant {tenant_id}",
    ),
]


class AlertManager:
    """Manages alert evaluation, storage, and retrieval."""

    def __init__(self):
        self._alerts: List[Alert] = []
        self._alert_counter = 0
        self._alerts_dir = Path("logs/alerts")
        self._alerts_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_rules(self, hours: int = 1) -> List[Alert]:
        """Evaluate all alert rules against current metrics."""
        collector = get_metrics_collector()
        new_alerts = []

        for tenant_id in ["sis", "mfg"]:
            metrics = collector.collect_tenant_metrics(tenant_id, hours)
            metrics_dict = asdict(metrics)

            for rule in ALERT_RULES:
                value = metrics_dict.get(rule.metric)
                if value is None:
                    continue

                # Skip rules if no data
                if metrics.total_requests == 0:
                    continue

                # Only check feedback if enough samples
                if rule.metric == "avg_feedback_rating" and metrics.feedback_count < 3:
                    continue

                triggered = False
                if rule.operator == "gt" and value > rule.threshold:
                    triggered = True
                elif rule.operator == "lt" and value < rule.threshold:
                    triggered = True
                elif rule.operator == "eq" and value == rule.threshold:
                    triggered = True

                if triggered:
                    self._alert_counter += 1
                    alert = Alert(
                        alert_id=f"alert_{self._alert_counter:04d}",
                        severity=rule.severity,
                        category=rule.category,
                        tenant_id=tenant_id,
                        message=rule.message_template.format(
                            value=value,
                            threshold=rule.threshold,
                            tenant_id=tenant_id,
                        ),
                        metric_name=rule.metric,
                        metric_value=value,
                        threshold=rule.threshold,
                        triggered_at=datetime.utcnow().isoformat(),
                    )
                    new_alerts.append(alert)
                    self._alerts.append(alert)

                    severity_icon = {"critical": "[CRIT]", "warning": "[WARN]", "info": "[INFO]"}
                    logger.warning(
                        f"{severity_icon.get(rule.severity, '[ALRT]')} "
                        f"[{rule.severity.upper()}] {alert.message}"
                    )

        if new_alerts:
            self._save_alerts(new_alerts)

        return new_alerts

    def get_active_alerts(self) -> List[Dict]:
        """Get all unresolved alerts."""
        return [asdict(a) for a in self._alerts if not a.resolved]

    def get_all_alerts(self, limit: int = 50) -> List[Dict]:
        """Get recent alerts."""
        return [asdict(a) for a in self._alerts[-limit:]]

    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                break

    def _save_alerts(self, alerts: List[Alert]):
        """Persist alerts to file."""
        filepath = self._alerts_dir / f"alerts_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
        with open(filepath, "a") as f:
            for alert in alerts:
                f.write(json.dumps(asdict(alert)) + "\n")


# Singleton
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
