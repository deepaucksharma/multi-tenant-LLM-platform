"""
Create tenant-specific Ollama model aliases for local development.

This script generates a Modelfile per tenant/model-type and runs
`ollama create` so the FastAPI app can route tenants to stable local
model names such as `tenant-sis-sft`.
"""
import argparse
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.tenant_router import TENANT_ROUTES

load_dotenv()

SUPPORTED_MODEL_TYPES = ("base", "sft", "dpo")


def resolve_source_model(model_type: str) -> str:
    return (
        os.getenv(f"OLLAMA_SOURCE_MODEL_{model_type.upper()}")
        or os.getenv("OLLAMA_SOURCE_MODEL")
        or os.getenv("OLLAMA_MODEL_DEFAULT")
        or os.getenv("OLLAMA_MODEL")
        or "qwen2.5:1.5b"
    )


def resolve_target_model(tenant_id: str, model_type: str) -> str:
    return (
        os.getenv(f"OLLAMA_MODEL_{tenant_id.upper()}_{model_type.upper()}")
        or os.getenv(f"OLLAMA_MODEL_{tenant_id.upper()}")
        or f"tenant-{tenant_id}-{model_type}"
    )


def build_system_prompt(tenant_id: str) -> str:
    route = TENANT_ROUTES[tenant_id]
    return route["system_prompt"]


def build_modelfile(source_model: str, tenant_id: str, adapter_path: str = "") -> str:
    lines = [
        f"FROM {source_model}",
        f'SYSTEM """{build_system_prompt(tenant_id)}"""',
    ]
    if adapter_path:
        lines.append(f"ADAPTER {adapter_path}")
    return "\n".join(lines) + "\n"


def create_model(target_model: str, modelfile_content: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False) as handle:
        handle.write(modelfile_content)
        modelfile_path = Path(handle.name)

    try:
        subprocess.run(
            ["ollama", "create", target_model, "-f", str(modelfile_path)],
            check=True,
        )
    finally:
        modelfile_path.unlink(missing_ok=True)


def iter_targets(tenant: str, model_type: str) -> Iterable[Tuple[str, str]]:
    tenants: List[str]
    model_types: List[str]

    tenants = [tenant] if tenant != "all" else list(TENANT_ROUTES.keys())
    model_types = [model_type] if model_type != "all" else list(SUPPORTED_MODEL_TYPES)

    for tenant_id in tenants:
        for variant in model_types:
            yield tenant_id, variant


def main() -> int:
    parser = argparse.ArgumentParser(description="Register tenant-specific Ollama models.")
    parser.add_argument(
        "--tenant",
        choices=["all", *TENANT_ROUTES.keys()],
        default="all",
        help="Tenant to register. Default: all",
    )
    parser.add_argument(
        "--model-type",
        choices=["all", *SUPPORTED_MODEL_TYPES],
        default="sft",
        help="Model type alias to create. Default: sft",
    )
    parser.add_argument(
        "--adapter-path",
        default="",
        help="Optional local adapter path to include in the Modelfile.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated model plan without calling `ollama create`.",
    )
    args = parser.parse_args()

    for tenant_id, model_type in iter_targets(args.tenant, args.model_type):
        source_model = resolve_source_model(model_type)
        target_model = resolve_target_model(tenant_id, model_type)
        modelfile_content = build_modelfile(source_model, tenant_id, args.adapter_path)

        print(f"[register] {tenant_id}/{model_type}: {target_model} <- {source_model}")
        if args.dry_run:
            print(modelfile_content)
            continue

        create_model(target_model, modelfile_content)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
