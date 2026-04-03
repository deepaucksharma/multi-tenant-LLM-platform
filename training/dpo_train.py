"""
Direct Preference Optimization (DPO) training script.
Aligns tenant models with domain-specific policies:
- SIS: FERPA compliance, privacy protection
- MFG: Safety-first, procedure adherence

Usage:
    python training/dpo_train.py --tenant sis
    python training/dpo_train.py --tenant mfg
    python training/dpo_train.py --tenant sis --sft-adapter ./models/adapters/sis/sft
    python training/dpo_train.py --tenant sis --full-reference
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

from training.config_loader import get_dpo_config, get_sft_config, deep_merge
from training.model_loader import (
    load_base_model_and_tokenizer,
    setup_lora,
    get_gpu_memory_info,
    get_training_runtime_config,
    resolve_model_source,
)
from training.data_loader import load_dpo_dataset
from training.mlflow_utils import ExperimentTracker, ModelRegistry


def train_dpo(
    tenant_id: str,
    sft_adapter_path: str = None,
    config_override: dict = None,
):
    """
    Run DPO training for a specific tenant.
    Starts from the SFT-adapted model if adapter path is provided.

    Args:
        tenant_id: "sis" or "mfg"
        sft_adapter_path: Path to SFT adapter to start from
        config_override: Optional config overrides
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"DPO TRAINING — TENANT: {tenant_id.upper()}")
    logger.info(f"{'='*70}")

    # Load configs
    dpo_config = get_dpo_config()
    sft_config = get_sft_config(tenant_id)

    # Merge base model config from SFT config
    config = {
        **sft_config,
        "dpo": dpo_config.get("dpo", {}),
        "training": dpo_config.get("training", {}),
    }
    if config_override:
        config = deep_merge(config, config_override)

    train_cfg = config["training"]
    model_cfg = config["model"]
    dpo_cfg = config["dpo"]
    runtime_cfg = get_training_runtime_config(config)

    # Output directory
    output_dir = f"./models/adapters/{tenant_id}/dpo"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Auto-detect SFT adapter if not provided
    if sft_adapter_path is None:
        default_sft = f"./models/adapters/{tenant_id}/sft"
        if Path(default_sft).exists() and (Path(default_sft) / "adapter_config.json").exists():
            sft_adapter_path = default_sft
            logger.info(f"Auto-detected SFT adapter: {sft_adapter_path}")
        else:
            logger.info("No SFT adapter found. Training DPO from base model.")

    # Initialize tracking
    tracker = ExperimentTracker(experiment_name=f"dpo-{tenant_id}")
    run_name = f"dpo_{tenant_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    tracker.start_run(
        run_name=run_name,
        tags={
            "tenant_id": tenant_id,
            "model_type": "dpo",
            "base_model": model_cfg["base_model"],
            "sft_adapter": sft_adapter_path or "none",
        },
    )

    try:
        gpu_info = get_gpu_memory_info()
        logger.info(f"GPU: {gpu_info}")
        tracker.log_params({"gpu": gpu_info})

        # ---- Load model ----
        logger.info("Loading model and tokenizer...")
        model, tokenizer = load_base_model_and_tokenizer(config, for_training=True)

        # ---- Load SFT adapter if available ----
        if sft_adapter_path:
            logger.info(f"Loading SFT adapter from {sft_adapter_path}...")
            from peft import PeftModel
            model = PeftModel.from_pretrained(
                model,
                sft_adapter_path,
                is_trainable=True,
            )
            # Need to merge and unload for DPO training with new LoRA
            logger.info("Merging SFT adapter for DPO training...")
            model = model.merge_and_unload()

        # ---- Apply fresh LoRA for DPO ----
        logger.info("Applying LoRA for DPO...")
        # Use smaller LoRA for DPO (less aggressive changes)
        dpo_lora_config = config["lora"].copy()
        dpo_lora_config["r"] = max(8, dpo_lora_config.get("r", 16) // 2)
        dpo_lora_config["lora_alpha"] = dpo_lora_config["r"] * 2
        config["lora"] = dpo_lora_config

        model, peft_config = setup_lora(model, config)

        # Log params
        tracker.log_params({
            "dpo_beta": dpo_cfg.get("beta", 0.1),
            "dpo_loss_type": dpo_cfg.get("loss_type", "sigmoid"),
            "device": runtime_cfg["device"],
            "use_bnb_4bit": runtime_cfg["use_bnb_4bit"],
            "dpo_lora_r": dpo_lora_config["r"],
            "learning_rate": train_cfg["learning_rate"],
            "num_epochs": train_cfg["num_train_epochs"],
            "batch_size": train_cfg["per_device_train_batch_size"],
            "sft_adapter": sft_adapter_path or "none",
        })

        # ---- Load DPO dataset ----
        logger.info("Loading DPO dataset...")
        dpo_train_path = f"./data/{tenant_id}/dpo/train_trl.json"
        dpo_eval_path = f"./data/{tenant_id}/dpo/eval_trl.json"

        train_dataset, eval_dataset = load_dpo_dataset(
            train_path=dpo_train_path,
            eval_path=dpo_eval_path,
            tokenizer=tokenizer,
        )

        tracker.log_params({
            "train_pairs": len(train_dataset),
            "eval_pairs": len(eval_dataset),
        })
        logger.info(f"DPO dataset: {len(train_dataset)} train, {len(eval_dataset)} eval")

        smoke_cfg = config.get("smoke_test", {})
        if smoke_cfg.get("enabled", False):
            train_limit = max(1, smoke_cfg.get("train_samples", 8))
            eval_limit = max(1, smoke_cfg.get("eval_samples", 4))
            train_dataset = train_dataset.select(range(min(len(train_dataset), train_limit)))
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), eval_limit)))
            smoke_seq_len = int(os.getenv("SMOKE_SEQ_LEN", "64"))
            dpo_cfg["max_prompt_length"] = min(dpo_cfg.get("max_prompt_length", 256), smoke_seq_len // 2)
            dpo_cfg["max_length"] = min(dpo_cfg.get("max_length", 512), smoke_seq_len)
            logger.info(
                f"Smoke test mode enabled: train={len(train_dataset)} eval={len(eval_dataset)}, "
                f"max_prompt_length={dpo_cfg['max_prompt_length']}, max_length={dpo_cfg['max_length']}"
            )

        # ---- Load reference model ----
        # DPO requires a frozen reference model alongside the policy model.
        # This doubles peak GPU/CPU memory usage.  Warn early if headroom is low.
        mem_info = get_gpu_memory_info()
        if mem_info.get("available") and mem_info.get("free_gb", 999) < 4.0:
            logger.warning(
                f"Low GPU memory ({mem_info['free_gb']:.1f} GB free) before loading "
                "reference model. DPO requires two model copies and may OOM. "
                "Consider using --smoke-test with a tiny model or reducing batch size."
            )
        logger.info("Loading reference model for DPO...")
        ref_model, _ = load_base_model_and_tokenizer(config, for_training=False)
        if sft_adapter_path:
            from peft import PeftModel
            ref_model = PeftModel.from_pretrained(
                ref_model,
                sft_adapter_path,
                is_trainable=False,
            )
            ref_model = ref_model.merge_and_unload()

        # ---- Configure Trainer ----
        from transformers import TrainingArguments
        from trl import DPOTrainer
        
        if runtime_cfg["device"] == "dml":
            import torch_directml
            logger.info("Monkeypatching HuggingFace Trainer device for DirectML...")
            TrainingArguments.device = property(lambda self: torch_directml.device())
            TrainingArguments.n_gpu = property(lambda self: 1)

        from trl import DPOConfig

        dpo_training_config = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=train_cfg["num_train_epochs"],
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            learning_rate=train_cfg["learning_rate"],
            weight_decay=train_cfg.get("weight_decay", 0.01),
            warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
            lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
            logging_steps=train_cfg.get("logging_steps", 2),
            eval_strategy=train_cfg.get("eval_strategy", "steps"),
            eval_steps=train_cfg.get("eval_steps", 10),
            save_strategy=train_cfg.get("save_strategy", "epoch"),
            save_total_limit=train_cfg.get("save_total_limit", 2),
            fp16=runtime_cfg["fp16"],
            bf16=runtime_cfg["bf16"],
            gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
            optim=runtime_cfg["optim"],
            max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
            seed=train_cfg.get("seed", 42),
            report_to="none",
            remove_unused_columns=False,
            beta=dpo_cfg.get("beta", 0.1),
            loss_type=dpo_cfg.get("loss_type", "sigmoid"),
            max_prompt_length=dpo_cfg.get("max_prompt_length", 256),
            max_length=dpo_cfg.get("max_length", 512),
            use_cpu=runtime_cfg["device"] == "cpu",
        )

        # ---- Create DPO Trainer ----
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=dpo_training_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )

        # ---- Train ----
        logger.info("Starting DPO training...")
        t_start = time.time()
        train_result = trainer.train()
        train_time = round(time.time() - t_start, 2)

        logger.info(f"DPO training completed in {train_time}s")

        # ---- Log metrics ----
        train_metrics = train_result.metrics
        tracker.log_metrics({
            "dpo_train_loss": train_metrics.get("train_loss", 0),
            "dpo_train_time_sec": train_time,
        })

        # Evaluate
        eval_metrics = trainer.evaluate()
        tracker.log_metrics({
            "dpo_eval_loss": eval_metrics.get("eval_loss", 0),
        })

        # ---- Save adapter ----
        logger.info(f"Saving DPO adapter to {output_dir}...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Save metadata
        metadata = {
            "tenant_id": tenant_id,
            "model_type": "dpo",
            "base_model": model_cfg["base_model"],
            "sft_adapter": sft_adapter_path,
            "adapter_path": output_dir,
            "dpo_config": dpo_cfg,
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "trained_at": datetime.utcnow().isoformat(),
            "train_time_sec": train_time,
        }
        metadata_path = Path(output_dir) / "training_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

        # ---- Register model ----
        registry = ModelRegistry()
        version = f"v{datetime.utcnow().strftime('%Y%m%d%H%M')}"
        registry.register_model(
            tenant_id=tenant_id,
            model_type="dpo",
            version=version,
            adapter_path=output_dir,
            base_model=model_cfg["base_model"],
            metrics={
                "dpo_train_loss": train_metrics.get("train_loss"),
                "dpo_eval_loss": eval_metrics.get("eval_loss"),
            },
            training_config=train_cfg,
            dataset_info={
                "train_pairs": len(train_dataset),
                "eval_pairs": len(eval_dataset),
            },
        )

        tracker.end_run(status="FINISHED")

        logger.info(f"\n{'='*70}")
        logger.info(f"DPO TRAINING COMPLETE — {tenant_id.upper()}")
        logger.info(f"Adapter saved: {output_dir}")
        logger.info(f"DPO train loss: {train_metrics.get('train_loss', 'N/A')}")
        logger.info(f"DPO eval loss: {eval_metrics.get('eval_loss', 'N/A')}")
        logger.info(f"Training time: {train_time}s")
        logger.info(f"{'='*70}")

        return {
            "status": "success",
            "tenant_id": tenant_id,
            "adapter_path": output_dir,
            "train_loss": train_metrics.get("train_loss"),
            "eval_loss": eval_metrics.get("eval_loss"),
            "train_time_sec": train_time,
        }

    except Exception as e:
        logger.error(f"DPO training failed: {e}")
        import traceback
        traceback.print_exc()
        tracker.end_run(status="FAILED")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="DPO Training")
    parser.add_argument(
        "--tenant", type=str, required=True,
        choices=["sis", "mfg"],
        help="Tenant to train for",
    )
    parser.add_argument(
        "--sft-adapter", type=str, default=None,
        help="Path to SFT adapter to start from",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--beta", type=float, default=None,
        help="Override DPO beta",
    )
    parser.add_argument(
        "--full-reference",
        action="store_true",
        help="Use the heavier explicit reference-model DPO path instead of the default simple mode",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run a tiny smoke-test DPO job with truncated datasets (CPU-safe)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Override max training steps",
    )
    args = parser.parse_args()

    overrides = {}
    if args.epochs:
        overrides.setdefault("training", {})["num_train_epochs"] = args.epochs
    if args.beta:
        overrides.setdefault("dpo", {})["beta"] = args.beta
    if args.max_steps is not None:
        overrides.setdefault("training", {})["max_steps"] = args.max_steps
    if args.smoke_test:
        max_steps = args.max_steps or 2
        overrides.setdefault("training", {})["num_train_epochs"] = 1
        overrides.setdefault("training", {})["max_steps"] = max_steps
        overrides.setdefault("training", {})["per_device_train_batch_size"] = 1
        overrides.setdefault("training", {})["per_device_eval_batch_size"] = 1
        overrides.setdefault("training", {})["gradient_accumulation_steps"] = 1
        overrides.setdefault("training", {})["logging_steps"] = 1
        overrides.setdefault("training", {})["save_total_limit"] = 1
        overrides["smoke_test"] = {
            "enabled": True,
            "train_samples": 8,
            "eval_samples": 4,
        }

    if args.full_reference:
        result = train_dpo(
            args.tenant,
            sft_adapter_path=args.sft_adapter,
            config_override=overrides if overrides else None,
        )
    else:
        from training.dpo_train_simple import train_dpo_simple

        logger.info("Using simplified DPO path by default to avoid reference-model OOMs")
        result = train_dpo_simple(
            args.tenant,
            sft_adapter_path=args.sft_adapter,
            config_override=overrides if overrides else None,
        )
    print(f"\nResult: {json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    main()
