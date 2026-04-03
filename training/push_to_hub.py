"""
Push trained LoRA adapters (and optional merged models) to Hugging Face Hub.

The Hub org/user is read from HF_HUB_NAMESPACE (default: deepaucksharma).
Each adapter is pushed as its own repository:

    deepaucksharma/multi-tenant-llm-sis-sft
    deepaucksharma/multi-tenant-llm-sis-dpo
    deepaucksharma/multi-tenant-llm-mfg-sft
    deepaucksharma/multi-tenant-llm-mfg-dpo

Usage:
    # Push a specific tenant + type
    python -m training.push_to_hub --tenant sis --type sft

    # Push all trained adapters
    python -m training.push_to_hub --all

    # Push and also create/update an eval card
    python -m training.push_to_hub --tenant sis --type sft --eval-report evaluation/reports/golden_set_sis_20260403_193000.json

    # Dry-run (prints what would happen, no upload)
    python -m training.push_to_hub --all --dry-run
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

HF_HUB_NAMESPACE: str = os.getenv("HF_HUB_NAMESPACE", "deepaucksharma")
HF_REPO_PREFIX: str = os.getenv("HF_REPO_PREFIX", "multi-tenant-llm")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")

# Adapter paths follow the convention models/adapters/{tenant}/{type}/
ADAPTERS_ROOT = Path("./models/adapters")

# Files that are safe to push (small metadata / config), even without --weights
METADATA_GLOBS = [
    "adapter_config.json",
    "training_metadata.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "generation_config.json",
    "config.json",
    "README.md",
]

# Binary weight files — only pushed when --weights flag is passed
WEIGHT_GLOBS = [
    "*.safetensors",
    "*.bin",
    "*.pt",
]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _repo_id(tenant_id: str, model_type: str) -> str:
    return f"{HF_HUB_NAMESPACE}/{HF_REPO_PREFIX}-{tenant_id}-{model_type}"


def _adapter_path(tenant_id: str, model_type: str) -> Path:
    return ADAPTERS_ROOT / tenant_id / model_type


def _load_training_metadata(adapter_dir: Path) -> dict:
    meta_path = adapter_dir / "training_metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def _build_model_card(
    tenant_id: str,
    model_type: str,
    metadata: dict,
    eval_report: Optional[dict] = None,
) -> str:
    """Generate a model card README.md for the Hub repo."""
    repo = _repo_id(tenant_id, model_type)
    base_model = metadata.get("base_model", "Qwen/Qwen2.5-1.5B-Instruct")
    trained_at = metadata.get("trained_at", datetime.now(timezone.utc).isoformat())
    domain_map = {
        "sis": "Student Information System / Education (District 42)",
        "mfg": "Manufacturing Quality Control & Operations",
    }
    domain = domain_map.get(tenant_id, tenant_id.upper())

    eval_section = ""
    if eval_report:
        scores = eval_report.get("scores", eval_report.get("results", {}))
        if scores:
            rows = "\n".join(
                f"| {k} | {v if isinstance(v, str) else f'{v:.1%}' if isinstance(v, float) and v <= 1 else v} |"
                for k, v in scores.items()
            )
            eval_section = f"""
## Evaluation Results

| Metric | Score |
|--------|-------|
{rows}

*Report generated: {eval_report.get('timestamp', trained_at)}*
"""

    return f"""---
base_model: {base_model}
library_name: peft
tags:
  - lora
  - peft
  - multi-tenant
  - {tenant_id}
  - {model_type}
  - llm-fine-tuning
license: apache-2.0
language:
  - en
---

# {repo.split('/')[-1]}

LoRA adapter fine-tuned from [{base_model}](https://huggingface.co/{base_model})
for the **{domain}** domain as part of the
[multi-tenant-LLM-platform](https://github.com/deepaucksharma/multi-tenant-LLM-platform).

## Details

| Field | Value |
|-------|-------|
| Base model | `{base_model}` |
| Training type | `{model_type.upper()}` |
| Tenant | `{tenant_id}` |
| Trained at | `{trained_at}` |
| PEFT method | LoRA |
{eval_section}

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base_model}")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")
model = PeftModel.from_pretrained(base, "{repo}")
```

## Source

- Repository: https://github.com/deepaucksharma/multi-tenant-LLM-platform
- Profile: https://huggingface.co/{HF_HUB_NAMESPACE}
"""


# ── Core push logic ───────────────────────────────────────────────────────────


def push_adapter(
    tenant_id: str,
    model_type: str,
    push_weights: bool = False,
    eval_report_path: Optional[str] = None,
    dry_run: bool = False,
) -> dict:
    """
    Push a single adapter to the Hub.

    Args:
        tenant_id:        'sis' or 'mfg'
        model_type:       'sft' or 'dpo'
        push_weights:     If True, also upload the .safetensors weight file.
                          Default False — only metadata + config.
        eval_report_path: Path to a JSON eval report to embed in the model card.
        dry_run:          Print actions without uploading.

    Returns:
        dict with 'repo_id', 'url', 'files_pushed'
    """
    adapter_dir = _adapter_path(tenant_id, model_type)
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_dir}\n"
            f"Run 'make train' first to produce {tenant_id}/{model_type} adapter."
        )

    repo_id = _repo_id(tenant_id, model_type)
    metadata = _load_training_metadata(adapter_dir)

    # Resolve eval report
    eval_report = None
    if eval_report_path:
        p = Path(eval_report_path)
        if p.exists():
            eval_report = json.loads(p.read_text(encoding="utf-8"))
        else:
            logger.warning(f"Eval report not found: {eval_report_path}")

    # Build model card
    model_card_text = _build_model_card(tenant_id, model_type, metadata, eval_report)

    # Collect files to push
    files_to_push: list[tuple[Path, str]] = []  # (local_path, path_in_repo)

    for glob in METADATA_GLOBS:
        for f in adapter_dir.glob(glob):
            files_to_push.append((f, f.name))

    if push_weights:
        for glob in WEIGHT_GLOBS:
            for f in adapter_dir.glob(glob):
                files_to_push.append((f, f.name))
        logger.info(f"[{repo_id}] Weight push enabled — including binary files")
    else:
        logger.info(
            f"[{repo_id}] Metadata-only push (use --weights to also push .safetensors)"
        )

    if dry_run:
        logger.info(f"[DRY RUN] Would create/update repo: {repo_id}")
        logger.info(f"[DRY RUN] Model card length: {len(model_card_text)} chars")
        for local, repo_path in files_to_push:
            size_kb = local.stat().st_size // 1024
            logger.info(f"[DRY RUN]   {repo_path}  ({size_kb} KB)")
        return {
            "repo_id": repo_id,
            "url": f"https://huggingface.co/{repo_id}",
            "files_pushed": [r for _, r in files_to_push],
            "dry_run": True,
        }

    # ── Real upload ──────────────────────────────────────────────────────────
    if not HF_TOKEN:
        raise EnvironmentError(
            "HF_TOKEN is not set. Add it to your .env file or environment.\n"
            "Get a write token from https://huggingface.co/settings/tokens"
        )

    from huggingface_hub import HfApi, RepoCard

    api = HfApi(token=HF_TOKEN)

    # Create repo if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=False,
        )
        logger.info(f"Repo ready: https://huggingface.co/{repo_id}")
    except Exception as exc:
        raise RuntimeError(f"Failed to create/verify repo '{repo_id}': {exc}") from exc

    # Push model card
    card = RepoCard(model_card_text)
    card.push_to_hub(repo_id, token=HF_TOKEN)
    logger.info(f"[{repo_id}] Model card pushed")

    # Push files
    pushed = []
    for local_path, repo_path in files_to_push:
        size_mb = local_path.stat().st_size / (1024 * 1024)
        logger.info(f"[{repo_id}] Uploading {repo_path} ({size_mb:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Update {repo_path} [{datetime.now(timezone.utc).strftime('%Y-%m-%d')}]",
        )
        pushed.append(repo_path)

    url = f"https://huggingface.co/{repo_id}"
    logger.success(f"[{repo_id}] Done → {url}")
    return {
        "repo_id": repo_id,
        "url": url,
        "files_pushed": pushed,
        "dry_run": False,
    }


def push_all(push_weights: bool = False, dry_run: bool = False) -> list[dict]:
    """Push all adapters that exist locally."""
    results = []
    if not ADAPTERS_ROOT.exists():
        logger.warning(f"Adapters root not found: {ADAPTERS_ROOT}  (run 'make train' first)")
        return results

    for tenant_dir in sorted(ADAPTERS_ROOT.iterdir()):
        if not tenant_dir.is_dir():
            continue
        tenant_id = tenant_dir.name
        for type_dir in sorted(tenant_dir.iterdir()):
            if not type_dir.is_dir():
                continue
            model_type = type_dir.name
            # Only push if adapter_config.json exists (confirms it's a real adapter)
            if not (type_dir / "adapter_config.json").exists():
                logger.debug(f"Skipping {tenant_id}/{model_type} — no adapter_config.json")
                continue
            try:
                result = push_adapter(
                    tenant_id, model_type, push_weights=push_weights, dry_run=dry_run
                )
                results.append(result)
            except Exception as exc:
                logger.error(f"Failed to push {tenant_id}/{model_type}: {exc}")
                results.append({"tenant": tenant_id, "type": model_type, "error": str(exc)})

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Push trained LoRA adapters to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--tenant", choices=["sis", "mfg"], help="Tenant to push")
    parser.add_argument("--type", dest="model_type", choices=["sft", "dpo"], default="sft")
    parser.add_argument(
        "--all", action="store_true", help="Push all adapters found under models/adapters/"
    )
    parser.add_argument(
        "--weights",
        action="store_true",
        help="Also upload .safetensors weight files (large — only do this intentionally)",
    )
    parser.add_argument(
        "--eval-report",
        type=str,
        default=None,
        help="Path to eval JSON report to embed in model card",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be pushed without uploading anything",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help=f"Override HF namespace (default: {HF_HUB_NAMESPACE})",
    )
    args = parser.parse_args()

    if args.namespace:
        global HF_HUB_NAMESPACE
        HF_HUB_NAMESPACE = args.namespace

    if args.all:
        results = push_all(push_weights=args.weights, dry_run=args.dry_run)
    elif args.tenant:
        results = [
            push_adapter(
                args.tenant,
                args.model_type,
                push_weights=args.weights,
                eval_report_path=args.eval_report,
                dry_run=args.dry_run,
            )
        ]
    else:
        parser.error("Specify --tenant <sis|mfg> or --all")
        return 1

    print(json.dumps(results, indent=2, default=str))
    errors = [r for r in results if "error" in r]
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
