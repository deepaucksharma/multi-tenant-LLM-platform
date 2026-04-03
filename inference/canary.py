"""
Canary deployment and A/B testing support.
Routes a percentage of traffic to a candidate model version.
"""
import json
import random
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from threading import Lock

from loguru import logger


class CanaryManager:
    """Manages canary/A/B deployment configuration per tenant."""

    def __init__(self):
        self._configs: Dict[str, Dict] = {}
        self._stats: Dict[str, Dict] = {}
        self._lock = Lock()
        self._load_config()

    def _load_config(self):
        """Load canary config from file if exists."""
        config_path = Path("models/canary_config.json")
        if config_path.exists():
            with open(config_path) as f:
                self._configs = json.load(f)
            logger.info(f"Canary config loaded: {list(self._configs.keys())}")

    def _save_config(self):
        """Save canary config to file."""
        config_path = Path("models/canary_config.json")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(self._configs, indent=2))

    def set_canary(
        self,
        tenant_id: str,
        stable_model: str = "sft",
        canary_model: str = "dpo",
        canary_percentage: float = 10.0,
        enabled: bool = True,
    ):
        """Configure canary deployment for a tenant."""
        with self._lock:
            self._configs[tenant_id] = {
                "stable_model": stable_model,
                "canary_model": canary_model,
                "canary_percentage": canary_percentage,
                "enabled": enabled,
                "created_at": datetime.utcnow().isoformat(),
            }
            self._stats[tenant_id] = {
                "stable_count": 0,
                "canary_count": 0,
                "total_count": 0,
            }
            self._save_config()
            logger.info(
                f"Canary configured for {tenant_id}: "
                f"{canary_percentage}% → {canary_model}, "
                f"{100-canary_percentage}% → {stable_model}"
            )

    def get_model_type(self, tenant_id: str) -> tuple:
        """
        Determine which model type to use for a request.

        Returns:
            (model_type, is_canary)
        """
        with self._lock:
            config = self._configs.get(tenant_id)
            if not config or not config.get("enabled", False):
                return "sft", False  # Default

            roll = random.random() * 100
            is_canary = roll < config["canary_percentage"]

            if tenant_id not in self._stats:
                self._stats[tenant_id] = {"stable_count": 0, "canary_count": 0, "total_count": 0}

            self._stats[tenant_id]["total_count"] += 1
            if is_canary:
                self._stats[tenant_id]["canary_count"] += 1
                return config["canary_model"], True
            else:
                self._stats[tenant_id]["stable_count"] += 1
                return config["stable_model"], False

    def disable_canary(self, tenant_id: str):
        """Disable canary for a tenant (rollback to stable)."""
        with self._lock:
            if tenant_id in self._configs:
                self._configs[tenant_id]["enabled"] = False
                self._save_config()
                logger.info(f"Canary disabled for {tenant_id} (rolled back to stable)")

    def promote_canary(self, tenant_id: str):
        """Promote canary to stable (full rollout)."""
        with self._lock:
            config = self._configs.get(tenant_id)
            if config:
                config["canary_percentage"] = 100.0
                self._save_config()
                logger.info(f"Canary promoted for {tenant_id}: 100% → {config['canary_model']}")

    def get_stats(self, tenant_id: str = None) -> Dict:
        """Get canary traffic statistics."""
        if tenant_id:
            return {
                "config": self._configs.get(tenant_id, {}),
                "stats": self._stats.get(tenant_id, {}),
            }
        return {
            "configs": self._configs,
            "stats": self._stats,
        }


# Global singleton
_canary_manager: Optional[CanaryManager] = None


def get_canary_manager() -> CanaryManager:
    global _canary_manager
    if _canary_manager is None:
        _canary_manager = CanaryManager()
    return _canary_manager
