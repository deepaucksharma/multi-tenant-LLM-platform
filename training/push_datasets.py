"""
Push SFT/DPO training datasets to Hugging Face Hub as dataset repositories.

Each tenant's JSONL files are uploaded without loading them fully into memory,
so this works even for large files on resource-constrained machines.

Dataset repos created:
    {namespace}/{prefix}-{tenant}-sft-data
    {namespace}/{prefix}-{tenant}-dpo-data

Usage:
    # Push all tenant datasets
    python -m training.push_datasets --all

    # Push a specific tenant + split
    python -m training.push_datasets --tenant sis --split sft

    # Dry-run (no upload)
    python -m training.push_datasets --all --dry-run

    # Push as private repos
    python -m training.push_datasets --all --private
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

# ── Config ─────────────────────────────────────────────────────────────────────

HF_HUB_NAMESPACE: str = os.getenv("HF_HUB_NAMESPACE", "deepaucksharma")
HF_DATASET_PREFIX: str = os.getenv("HF_DATASET_PREFIX", os.getenv("HF_REPO_PREFIX", "multi-tenant-llm"))
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")

DATA_ROOT = Path(os.getenv("DATA_ROOT", "./data"))
TENANTS = ["sis", "mfg"]
SPLITS = ["sft", "dpo"]

DOMAIN_MAP = {
    "sis": "Student Information System / Education (District 42)",
    "mfg": "Manufacturing Quality Control & Operations",
}


# ── Helpers ────────────────────────────────────────────────────────────────────


def _repo_id(tenant_id: str, split: str) -> str:
    return f"{HF_HUB_NAMESPACE}/{HF_DATASET_PREFIX}-{tenant_id}-{split}-data"


def _dataset_dir(tenant_id: str, split: str) -> Path:
    return DATA_ROOT / tenant_id / split


def _count_records(path: Path) -> int:
    """Count records in a .json (JSON array) or .jsonl file."""
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return len(data) if isinstance(data, list) else 1
    # .jsonl: count non-empty lines
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _infer_schema(path: Path) -> dict:
    """Read first record to infer field names."""
    try:
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return data[0] if isinstance(data, list) and data else {}
        # .jsonl: first non-empty line
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    return json.loads(line)
    except (json.JSONDecodeError, IndexError):
        pass
    return {}


def _build_dataset_card(
    tenant_id: str,
    split: str,
    files: list[tuple[Path, int]],
    generated_at: str,
) -> str:
    repo = _repo_id(tenant_id, split)
    domain = DOMAIN_MAP.get(tenant_id, tenant_id.upper())
    total = sum(n for _, n in files)

    field_names: list[str] = []
    if files:
        sample = _infer_schema(files[0][0])
        field_names = list(sample.keys())

    schema_rows = "\n".join(f"| `{f}` | string |" for f in field_names) if field_names else "| (unknown) | — |"

    file_rows = "\n".join(
        f"| `{p.name}` | {n:,} |" for p, n in files
    )

    return f"""---
task_categories:
  - text-generation
language:
  - en
tags:
  - multi-tenant
  - {tenant_id}
  - {split}
  - llm-fine-tuning
license: apache-2.0
---

# {repo.split('/')[-1]}

{split.upper()} training dataset for the **{domain}** tenant, part of the
[multi-tenant-LLM-platform](https://github.com/deepaucksharma/multi-tenant-LLM-platform).

## Details

| Field | Value |
|-------|-------|
| Tenant | `{tenant_id}` |
| Split | `{split.upper()}` |
| Total examples | {total:,} |
| Generated | `{generated_at}` |

## Schema

| Field | Type |
|-------|------|
{schema_rows}

## Files

| File | Examples |
|------|----------|
{file_rows}

## Source

- Repository: https://github.com/deepaucksharma/multi-tenant-LLM-platform
- Profile: https://huggingface.co/{HF_HUB_NAMESPACE}
"""


# ── Core push logic ────────────────────────────────────────────────────────────


def push_dataset(
    tenant_id: str,
    split: str,
    private: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Push JSONL files for one (tenant, split) pair to HF Hub as a dataset repo.

    Returns dict with repo_id, url, files_pushed, sample_counts, dry_run.
    """
    dataset_dir = _dataset_dir(tenant_id, split)
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            f"Run 'make data' first to generate {tenant_id}/{split} data."
        )

    data_files = sorted(
        list(dataset_dir.glob("*.json")) + list(dataset_dir.glob("*.jsonl"))
    )
    if not data_files:
        raise FileNotFoundError(
            f"No .json or .jsonl files found in {dataset_dir}"
        )

    repo_id = _repo_id(tenant_id, split)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Count records per file
    files_with_counts: list[tuple[Path, int]] = []
    for f in data_files:
        n = _count_records(f)
        files_with_counts.append((f, n))
        logger.debug(f"  {f.name}: {n:,} examples")

    dataset_card_text = _build_dataset_card(tenant_id, split, files_with_counts, generated_at)
    total = sum(n for _, n in files_with_counts)

    if dry_run:
        logger.info(f"[DRY RUN] Would create/update dataset repo: {repo_id}")
        logger.info(f"[DRY RUN] Private: {private}")
        logger.info(f"[DRY RUN] Dataset card: {len(dataset_card_text)} chars")
        for f, n in files_with_counts:
            size_kb = f.stat().st_size // 1024
            logger.info(f"[DRY RUN]   {f.name}  ({n:,} rows, {size_kb} KB)")
        return {
            "repo_id": repo_id,
            "url": f"https://huggingface.co/datasets/{repo_id}",
            "files_pushed": [f.name for f, _ in files_with_counts],
            "sample_counts": {f.name: n for f, n in files_with_counts},
            "total_examples": total,
            "dry_run": True,
        }

    # ── Real upload ────────────────────────────────────────────────────────────
    if not HF_TOKEN:
        raise EnvironmentError(
            "HF_TOKEN is not set. Add it to your .env file or environment.\n"
            "Get a write token from https://huggingface.co/settings/tokens"
        )

    from huggingface_hub import HfApi, DatasetCard

    api = HfApi(token=HF_TOKEN)

    # Create repo
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=private,
        )
        logger.info(f"Dataset repo ready: https://huggingface.co/datasets/{repo_id}")
    except Exception as exc:
        raise RuntimeError(f"Failed to create dataset repo '{repo_id}': {exc}") from exc

    # Push dataset card
    card = DatasetCard(dataset_card_text)
    card.push_to_hub(repo_id, token=HF_TOKEN)
    logger.info(f"[{repo_id}] Dataset card pushed")

    # Upload JSONL files
    pushed = []
    for local_path, n_rows in files_with_counts:
        size_mb = local_path.stat().st_size / (1024 * 1024)
        logger.info(f"[{repo_id}] Uploading {local_path.name} ({n_rows:,} rows, {size_mb:.1f} MB)…")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=f"data/{local_path.name}",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Update {local_path.name} [{generated_at}]",
        )
        pushed.append(local_path.name)

    url = f"https://huggingface.co/datasets/{repo_id}"
    logger.success(f"[{repo_id}] Done → {url}")
    return {
        "repo_id": repo_id,
        "url": url,
        "files_pushed": pushed,
        "sample_counts": {f.name: n for f, n in files_with_counts},
        "total_examples": total,
        "dry_run": False,
    }


def push_all_datasets(private: bool = False, dry_run: bool = False) -> list[dict]:
    """Push all tenant/split datasets that exist locally."""
    results = []
    for tenant_id in TENANTS:
        for split in SPLITS:
            dataset_dir = _dataset_dir(tenant_id, split)
            has_files = dataset_dir.exists() and (
                list(dataset_dir.glob("*.json")) or list(dataset_dir.glob("*.jsonl"))
            )
            if not has_files:
                logger.debug(f"Skipping {tenant_id}/{split} — no data files found")
                continue
            try:
                result = push_dataset(tenant_id, split, private=private, dry_run=dry_run)
                results.append(result)
            except Exception as exc:
                logger.error(f"Failed to push {tenant_id}/{split}: {exc}")
                results.append({"tenant": tenant_id, "split": split, "error": str(exc)})
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Push SFT/DPO training datasets to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--tenant", choices=TENANTS, help="Specific tenant to push")
    parser.add_argument("--split", choices=SPLITS, default="sft", help="Dataset split (default: sft)")
    parser.add_argument("--all", action="store_true", help="Push all tenants × splits")
    parser.add_argument("--private", action="store_true", help="Create private HF repos (default: public)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be pushed, no upload")
    parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help=f"Override HF namespace (default: {HF_HUB_NAMESPACE})",
    )
    args = parser.parse_args()

    if args.namespace:
        import training.push_datasets as _self
        _self.HF_HUB_NAMESPACE = args.namespace

    if args.all:
        results = push_all_datasets(private=args.private, dry_run=args.dry_run)
    elif args.tenant:
        results = [push_dataset(args.tenant, args.split, private=args.private, dry_run=args.dry_run)]
    else:
        parser.error("Specify --tenant <sis|mfg> or --all")
        return 1

    print(json.dumps(results, indent=2, default=str))
    errors = [r for r in results if "error" in r]
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
