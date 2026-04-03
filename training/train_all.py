"""
Orchestrator script to train all tenants sequentially.
Handles the full training lifecycle:
  1. SFT for tenant A
  2. SFT for tenant B
  3. DPO for tenant A (starting from SFT)
  4. DPO for tenant B (starting from SFT)

Usage:
    python training/train_all.py
    python training/train_all.py --skip-dpo
    python training/train_all.py --tenants sis
"""
import argparse
import json
import time
import gc
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger

from training.sft_train import train_sft
from training.mlflow_utils import ModelRegistry


def clear_gpu():
    """Force GPU memory cleanup between training runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.info("GPU memory cleared")


def train_all(
    tenants: list = None,
    skip_dpo: bool = False,
    sft_epochs: int = None,
    dpo_epochs: int = None,
):
    """
    Run full training pipeline for all tenants.
    """
    tenants = tenants or ["sis", "mfg"]

    logger.info(f"\n{'='*70}")
    logger.info(f"FULL TRAINING PIPELINE")
    logger.info(f"Tenants: {tenants}")
    logger.info(f"DPO: {'skip' if skip_dpo else 'enabled'}")
    logger.info(f"{'='*70}")

    results = {
        "started_at": datetime.utcnow().isoformat(),
        "tenants": tenants,
        "sft_results": {},
        "dpo_results": {},
    }

    t_start = time.time()

    # ---- Phase 1: SFT Training ----
    logger.info("\nPHASE 1: Supervised Fine-Tuning")
    for tenant_id in tenants:
        logger.info(f"\n--- SFT: {tenant_id.upper()} ---")

        overrides = {}
        if sft_epochs:
            overrides["training"] = {"num_train_epochs": sft_epochs}

        try:
            result = train_sft(tenant_id, config_override=overrides if overrides else None)
            results["sft_results"][tenant_id] = result
            logger.info(f"SFT {tenant_id}: success | train_loss={result.get('train_loss')}")
        except Exception as e:
            logger.error(f"SFT {tenant_id}: failed | {e}")
            results["sft_results"][tenant_id] = {"status": "failed", "error": str(e)}

        clear_gpu()

    # ---- Phase 2: DPO Training ----
    if not skip_dpo:
        logger.info("\nPHASE 2: DPO Preference Alignment")
        for tenant_id in tenants:
            logger.info(f"\n--- DPO: {tenant_id.upper()} ---")

            try:
                # Use simplified DPO to save memory
                from training.dpo_train_simple import train_dpo_simple
                result = train_dpo_simple(tenant_id)
                results["dpo_results"][tenant_id] = result
                logger.info(f"DPO {tenant_id}: success")
            except Exception as e:
                logger.error(f"DPO {tenant_id}: failed | {e}")
                results["dpo_results"][tenant_id] = {"status": "failed", "error": str(e)}

            clear_gpu()

    # ---- Summary ----
    total_time = round(time.time() - t_start, 2)
    results["total_time_sec"] = total_time
    results["completed_at"] = datetime.utcnow().isoformat()

    # Promote best models to production
    registry = ModelRegistry()
    for tenant_id in tenants:
        tenant_models = registry.list_models(tenant_id)
        for model_type in ["dpo", "sft"]:
            matching = [m for m in tenant_models if m["model_type"] == model_type and m.get("status") != "failed"]
            if matching:
                latest = matching[-1]
                registry.promote_to_production(tenant_id, model_type, latest["version"])
                logger.info(f"Promoted {tenant_id}/{model_type}/{latest['version']} to production")
                break

    # Save results
    report_dir = Path("evaluation/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"training_run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(results, indent=2, default=str))

    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING PIPELINE COMPLETE")
    logger.info(f"Total time: {total_time}s")
    logger.info(f"Report: {report_path}")
    logger.info(f"\nModel Registry:")
    summary = registry.get_summary()
    logger.info(f"  Total models: {summary['total_models']}")
    logger.info(f"  Active versions: {summary['active_versions']}")
    logger.info(f"{'='*70}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Full training pipeline")
    parser.add_argument("--tenants", nargs="+", default=None, choices=["sis", "mfg"])
    parser.add_argument("--skip-dpo", action="store_true")
    parser.add_argument("--sft-epochs", type=int, default=None)
    parser.add_argument("--dpo-epochs", type=int, default=None)
    args = parser.parse_args()

    train_all(
        tenants=args.tenants,
        skip_dpo=args.skip_dpo,
        sft_epochs=args.sft_epochs,
        dpo_epochs=args.dpo_epochs,
    )


if __name__ == "__main__":
    main()
