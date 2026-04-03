"""
Model rollback support.
Allows reverting to a previous model version.
"""
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

from loguru import logger


def list_versions(tenant_id: str, model_type: str = "sft") -> List[Dict]:
    """List all available versions for a tenant model."""
    registry_path = Path("models/registry.json")
    if not registry_path.exists():
        return []

    with open(registry_path) as f:
        registry = json.load(f)

    versions = [
        m for m in registry.get("models", [])
        if m["tenant_id"] == tenant_id and m["model_type"] == model_type
    ]

    return sorted(versions, key=lambda m: m.get("registered_at", ""), reverse=True)


def get_current_version(tenant_id: str, model_type: str = "sft") -> Optional[Dict]:
    """Get the current production version."""
    registry_path = Path("models/registry.json")
    if not registry_path.exists():
        return None

    with open(registry_path) as f:
        registry = json.load(f)

    active_key = f"{tenant_id}_{model_type}"
    active_id = registry.get("active_versions", {}).get(active_key)

    if active_id:
        for m in registry.get("models", []):
            if m["id"] == active_id:
                return m

    return None


def rollback(
    tenant_id: str,
    model_type: str = "sft",
    target_version: str = None,
) -> Dict:
    """
    Rollback to a previous model version.

    Args:
        tenant_id: Tenant to rollback
        model_type: Model type (sft, dpo)
        target_version: Specific version to rollback to, or None for previous

    Returns:
        Rollback result dict
    """
    registry_path = Path("models/registry.json")
    if not registry_path.exists():
        return {"status": "error", "message": "No registry found"}

    with open(registry_path) as f:
        registry = json.load(f)

    versions = list_versions(tenant_id, model_type)
    if len(versions) < 2:
        return {"status": "error", "message": "Not enough versions for rollback"}

    current = get_current_version(tenant_id, model_type)
    current_version = current["version"] if current else "none"

    if target_version:
        target = next((v for v in versions if v["version"] == target_version), None)
        if not target:
            return {"status": "error", "message": f"Version {target_version} not found"}
    else:
        target = None
        for v in versions:
            if v.get("version") != current_version:
                target = v
                break
        if not target:
            return {"status": "error", "message": "No previous version found"}

    # Perform rollback
    target_id = target["id"]
    active_key = f"{tenant_id}_{model_type}"

    # Demote current
    if current:
        for m in registry["models"]:
            if m["id"] == current["id"]:
                m["status"] = "rolled_back"
                m["rolled_back_at"] = datetime.utcnow().isoformat()

    # Promote target
    for m in registry["models"]:
        if m["id"] == target_id:
            m["status"] = "production"
            m["promoted_at"] = datetime.utcnow().isoformat()
            m["rollback_from"] = current_version

    registry["active_versions"][active_key] = target_id

    # Save
    registry_path.write_text(json.dumps(registry, indent=2))

    result = {
        "status": "success",
        "tenant_id": tenant_id,
        "model_type": model_type,
        "rolled_back_from": current_version,
        "rolled_back_to": target["version"],
        "adapter_path": target["adapter_path"],
        "timestamp": datetime.utcnow().isoformat(),
    }

    logger.info(
        f"ROLLBACK: {tenant_id}/{model_type} "
        f"from {current_version} to {target['version']}"
    )

    # Save rollback log
    rollback_log = Path("logs/rollbacks.jsonl")
    rollback_log.parent.mkdir(parents=True, exist_ok=True)
    with open(rollback_log, "a") as f:
        f.write(json.dumps(result) + "\n")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model rollback")
    parser.add_argument("--tenant", required=True, choices=["sis", "mfg"])
    parser.add_argument("--type", default="sft", choices=["sft", "dpo"])
    parser.add_argument("--version", default=None, help="Target version")
    parser.add_argument("--list", action="store_true", help="List versions only")
    args = parser.parse_args()

    if args.list:
        versions = list_versions(args.tenant, args.type)
        current = get_current_version(args.tenant, args.type)
        print(f"\nVersions for {args.tenant}/{args.type}:")
        for v in versions:
            marker = " <- current" if current and v["id"] == current["id"] else ""
            print(f"  {v['version']} [{v['status']}] loss={v.get('metrics', {}).get('eval_loss', 'N/A')}{marker}")
    else:
        result = rollback(args.tenant, args.type, args.version)
        print(json.dumps(result, indent=2))
