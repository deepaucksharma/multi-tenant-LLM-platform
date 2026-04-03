"""
Check whether a training base model is available locally or likely requires download.

Run as:
    python -m training.check_model --tenant sis [--smoke-test]
"""
import argparse
import json
import os
from pathlib import Path

from training.config_loader import get_sft_config
from training.model_loader import resolve_model_source


def build_report(tenant: str, smoke_test: bool) -> dict:
    cfg = get_sft_config(tenant)
    if smoke_test:
        cfg["smoke_test"] = {"enabled": True}

    resolved = resolve_model_source(cfg, smoke_test=smoke_test)
    resolved_path = Path(resolved)
    is_local_dir = resolved_path.exists()
    has_files = is_local_dir and any(resolved_path.iterdir())

    return {
        "tenant": tenant,
        "smoke_test": smoke_test,
        "resolved_model_source": resolved,
        "is_local_directory": is_local_dir,
        "local_directory_has_files": has_files,
        "ready_without_network": bool(has_files),
        "hint": (
            "Model files are available locally."
            if has_files
            else "Provide SMOKE_TEST_LOCAL_MODEL_PATH or populate models/base before offline training."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check training model availability")
    parser.add_argument("--tenant", choices=["sis", "mfg"], default="sis")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    report = build_report(args.tenant, args.smoke_test)
    print(json.dumps(report, indent=2))
    return 0 if report["ready_without_network"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
