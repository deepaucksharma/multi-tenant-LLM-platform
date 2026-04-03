"""
Generate a Google Colab-ready training notebook: notebooks/train_on_colab.ipynb

The notebook pulls datasets from HF Hub, runs SFT/DPO training on a free T4 GPU,
and pushes the trained adapter back to HF Hub — all without local GPU.

Usage:
    python notebooks/generate_colab_notebook.py
    # → writes notebooks/train_on_colab.ipynb

Open the generated notebook on Google Colab:
    https://colab.research.google.com/  →  File → Upload notebook
"""
import json
import sys
from pathlib import Path


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip(),
    }


def _markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip(),
    }


CELLS = [
    _markdown("""# Multi-Tenant LLM Platform — Colab Training

Train LoRA adapters for the SIS and MFG tenants on a **free T4 GPU** in Google Colab,
then push the trained adapters back to Hugging Face Hub.

> **Before running:** Select *Runtime → Change runtime type → T4 GPU* in Colab.
"""),

    # ── Cell 1: Install dependencies ──────────────────────────────────────────
    _markdown("## 1. Install Dependencies"),
    _code("""\
%%capture
!pip install -q peft trl transformers accelerate datasets bitsandbytes huggingface_hub loguru python-dotenv mlflow
print("✓ Dependencies installed")
"""),

    # ── Cell 2: Clone repo + HF login ─────────────────────────────────────────
    _markdown("## 2. Clone Repository & Authenticate"),
    _code("""\
import os

# Clone the repo (replace with your fork if needed)
REPO_URL = "https://github.com/deepaucksharma/multi-tenant-LLM-platform"
if not os.path.exists("multi-tenant-LLM-platform"):
    !git clone --depth 1 {REPO_URL}

%cd multi-tenant-LLM-platform

# Authenticate with HF Hub (required for pushing adapters)
from huggingface_hub import notebook_login
notebook_login()
"""),

    # ── Cell 3: Pull datasets from HF Hub ─────────────────────────────────────
    _markdown("## 3. Pull Datasets from Hugging Face Hub"),
    _code("""\
import os
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

HF_NAMESPACE = "deepaucksharma"
HF_PREFIX    = "multi-tenant-llm"

def pull_dataset(tenant: str, split: str):
    repo_id = f"{HF_NAMESPACE}/{HF_PREFIX}-{tenant}-{split}-data"
    out_dir = Path(f"data/{tenant}/{split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        files = [f for f in list_repo_files(repo_id, repo_type="dataset") if f.startswith("data/") and f.endswith(".jsonl")]
        for repo_path in files:
            local = hf_hub_download(repo_id=repo_id, filename=repo_path, repo_type="dataset")
            dest = out_dir / Path(repo_path).name
            import shutil; shutil.copy(local, dest)
            print(f"  ✓ {repo_id}/{repo_path} → {dest}")
    except Exception as e:
        print(f"  ✗ Could not pull {repo_id}: {e} (will generate locally instead)")

for tenant in ("sis", "mfg"):
    for split in ("sft", "dpo"):
        print(f"Pulling {tenant}/{split}…")
        pull_dataset(tenant, split)

# Fall back to generating data locally if Hub datasets don't exist yet
if not list(Path("data/sis/sft").glob("*.jsonl")):
    print("\\nGenerating data locally (Hub datasets not found)…")
    !python -m tenant_data_pipeline.run_pipeline
"""),

    # ── Cell 4: Run SFT training ───────────────────────────────────────────────
    _markdown("## 4. Train SFT Adapters\n\nSet `SMOKE_TEST=True` for a quick validation run (~5 min), or `False` for full training (~1-2 hr on T4)."),
    _code("""\
SMOKE_TEST = True   # ← set False for full training
TENANT     = "sis"  # ← "sis" or "mfg"

smoke_flag = "--smoke-test" if SMOKE_TEST else ""
print(f"Training {TENANT} SFT {'(smoke test)' if SMOKE_TEST else '(full)'}…")
!python -m training.sft_train --tenant {TENANT} {smoke_flag}
print("✓ SFT training complete")
"""),

    # ── Cell 5: Push adapter to Hub ────────────────────────────────────────────
    _markdown("## 5. Push Trained Adapter to Hugging Face Hub"),
    _code("""\
TENANT     = "sis"
MODEL_TYPE = "sft"
PUSH_WEIGHTS = True  # set False to push metadata only (faster, no large files)

weights_flag = "--weights" if PUSH_WEIGHTS else ""
print(f"Pushing {TENANT}/{MODEL_TYPE} adapter to Hub…")
!python -m training.push_to_hub --tenant {TENANT} --type {MODEL_TYPE} {weights_flag}
print("✓ Adapter pushed")
"""),

    # ── Cell 6: Run evaluation + push report ───────────────────────────────────
    _markdown("## 6. Evaluate & Push Eval Report (Optional)"),
    _code("""\
import glob, os

# Run golden-set evaluation (requires inference server or local HF backend)
TENANT = "sis"
print("Running evaluation…")
!python -m evaluation.run_all_evals --tenant {TENANT} 2>&1 | tail -20

# Find the latest report
reports = sorted(glob.glob(f"evaluation/reports/golden_set_{TENANT}_*.json"))
if reports:
    latest = reports[-1]
    print(f"\\nLatest report: {latest}")
    !python -m training.push_to_hub --tenant {TENANT} --type sft --eval-report {latest}
    print("✓ Eval report pushed to Hub")
else:
    print("No eval report found — skipping push")
"""),
]


def build_notebook(cells: list) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
            "accelerator": "GPU",
            "colab": {
                "provenance": [],
                "gpuType": "T4",
            },
        },
        "cells": cells,
    }


def main():
    out_path = Path(__file__).parent / "train_on_colab.ipynb"
    nb = build_notebook(CELLS)
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Generated: {out_path}")
    print("Open in Colab: https://colab.research.google.com/ → File → Upload notebook")


if __name__ == "__main__":
    main()
