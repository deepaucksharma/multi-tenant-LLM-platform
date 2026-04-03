"""
Model registry CLI interface.
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from loguru import logger


class ModelRegistry:
    """Simple file-based model registry."""

    def __init__(self, registry_path: str = "models/registry.json"):
        self.registry_path = Path(registry_path)
        self._ensure_registry()

    def _ensure_registry(self):
        """Create registry file if it doesn't exist."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self.registry_path.write_text(json.dumps({
                "models": [],
                "active_versions": {},
                "created_at": datetime.utcnow().isoformat(),
            }, indent=2))

    def _load(self) -> Dict:
        with open(self.registry_path) as f:
            return json.load(f)

    def _save(self, registry: Dict):
        self.registry_path.write_text(json.dumps(registry, indent=2))

    def register_model(
        self,
        tenant_id: str,
        model_type: str,
        adapter_path: str,
        metrics: Dict,
        version: str = None,
        status: str = "staging",
    ) -> str:
        """Register a new model version."""
        registry = self._load()

        if version is None:
            version = f"v{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        model_id = f"{tenant_id}_{model_type}_{version}"

        entry = {
            "id": model_id,
            "tenant_id": tenant_id,
            "model_type": model_type,
            "adapter_path": adapter_path,
            "version": version,
            "status": status,
            "metrics": metrics,
            "registered_at": datetime.utcnow().isoformat(),
        }

        registry["models"].append(entry)
        self._save(registry)

        logger.info(f"Registered model: {model_id} [{status}]")
        return model_id

    def promote_to_production(self, tenant_id: str, model_type: str, version: str):
        """Promote a model version to production."""
        registry = self._load()
        active_key = f"{tenant_id}_{model_type}"

        # Demote current production
        for m in registry["models"]:
            if (m["tenant_id"] == tenant_id and
                    m["model_type"] == model_type and
                    m["status"] == "production"):
                m["status"] = "archived"

        # Promote target
        for m in registry["models"]:
            if (m["tenant_id"] == tenant_id and
                    m["model_type"] == model_type and
                    m["version"] == version):
                m["status"] = "production"
                m["promoted_at"] = datetime.utcnow().isoformat()
                registry["active_versions"][active_key] = m["id"]
                break

        self._save(registry)
        logger.info(f"Promoted {tenant_id}/{model_type}/{version} to production")

    def list_models(self, tenant_id: str = None) -> List[Dict]:
        """List all registered models."""
        registry = self._load()
        models = registry.get("models", [])
        if tenant_id:
            models = [m for m in models if m["tenant_id"] == tenant_id]
        return sorted(models, key=lambda m: m.get("registered_at", ""), reverse=True)

    def get_active_model(self, tenant_id: str, model_type: str) -> Optional[Dict]:
        """Get the current active model for a tenant."""
        registry = self._load()
        active_key = f"{tenant_id}_{model_type}"
        active_id = registry.get("active_versions", {}).get(active_key)

        if active_id:
            for m in registry.get("models", []):
                if m["id"] == active_id:
                    return m
        return None

    def get_summary(self) -> Dict:
        """Get registry summary."""
        registry = self._load()
        models = registry.get("models", [])

        by_tenant = {}
        for m in models:
            tid = m["tenant_id"]
            if tid not in by_tenant:
                by_tenant[tid] = {"total": 0, "production": 0, "staging": 0}
            by_tenant[tid]["total"] += 1
            status = m.get("status", "")
            if status in by_tenant[tid]:
                by_tenant[tid][status] += 1

        return {
            "total_models": len(models),
            "active_versions": registry.get("active_versions", {}),
            "by_tenant": by_tenant,
        }


def main():
    parser = argparse.ArgumentParser(description="Model Registry CLI")
    parser.add_argument("action", choices=["list", "summary", "promote", "history"])
    parser.add_argument("--tenant", type=str, default=None)
    parser.add_argument("--type", type=str, default="sft")
    parser.add_argument("--version", type=str, default=None)
    args = parser.parse_args()

    registry = ModelRegistry()

    if args.action == "list":
        models = registry.list_models(args.tenant)
        if not models:
            print("No models registered.")
            return
        print(f"\n{'ID':<35} {'Tenant':<6} {'Type':<5} {'Version':<20} {'Status':<12} {'Metrics'}")
        print("-" * 110)
        for m in models:
            metrics_str = json.dumps(m.get("metrics", {}))[:30]
            print(
                f"{m['id']:<35} {m['tenant_id']:<6} {m['model_type']:<5} "
                f"{m['version']:<20} {m['status']:<12} {metrics_str}"
            )

    elif args.action == "summary":
        summary = registry.get_summary()
        print(f"\nTotal models: {summary['total_models']}")
        print(f"Active versions: {json.dumps(summary['active_versions'], indent=2)}")
        if summary.get('by_tenant'):
            print(f"By tenant: {json.dumps(summary['by_tenant'], indent=2)}")

    elif args.action == "promote":
        if not args.tenant or not args.version:
            print("Error: --tenant and --version required for promote")
            return
        registry.promote_to_production(args.tenant, args.type, args.version)
        print(f"Promoted {args.tenant}/{args.type}/{args.version} to production")

    elif args.action == "history":
        models = registry.list_models(args.tenant)
        for m in models:
            status_icon = {"production": "[PROD]", "staging": "[STAG]", "archived": "[ARCH]", "rolled_back": "[BACK]"}
            icon = status_icon.get(m.get("status", ""), "[----]")
            print(
                f"{icon} {m['version']} [{m['status']}] "
                f"registered={m.get('registered_at', 'N/A')[:19]} "
                f"tenant={m['tenant_id']} type={m['model_type']}"
            )


if __name__ == "__main__":
    main()
