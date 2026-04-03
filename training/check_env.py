"""
Training environment check for adaptive local execution.

This validates that the required Python packages for training are installed
and reports which runtime path will be used on the current machine.
"""
import importlib.util
import json

from training.config_loader import get_sft_config
from training.model_loader import get_training_runtime_config, resolve_device


REQUIRED_PACKAGES = [
    "torch",
    "transformers",
    "peft",
    "trl",
    "datasets",
    "yaml",
]

OPTIONAL_PACKAGES = [
    "bitsandbytes",
    "mlflow",
]


def is_installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    package_status = {name: is_installed(name) for name in REQUIRED_PACKAGES}
    optional_status = {name: is_installed(name) for name in OPTIONAL_PACKAGES}
    missing = [name for name, present in package_status.items() if not present]

    cfg = get_sft_config("sis")
    runtime = get_training_runtime_config(cfg)

    report = {
        "device": resolve_device(),
        "runtime": {
            "device": runtime["device"],
            "use_bnb_4bit": runtime["use_bnb_4bit"],
            "optim": runtime["optim"],
            "fp16": runtime["fp16"],
            "bf16": runtime["bf16"],
            "torch_dtype": str(runtime["torch_dtype"]).replace("torch.", ""),
        },
        "required_packages": package_status,
        "optional_packages": optional_status,
        "ready_for_training": len(missing) == 0,
        "missing_required": missing,
    }

    print(json.dumps(report, indent=2))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
