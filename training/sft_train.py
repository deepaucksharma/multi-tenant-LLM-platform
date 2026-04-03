"""
Supervised Fine-Tuning (SFT) training script.
Uses QLoRA on Qwen2.5-1.5B-Instruct for tenant-specific domain adaptation.

Usage:
    python training/sft_train.py --tenant sis
    python training/sft_train.py --tenant mfg
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

from training.config_loader import get_sft_config
from training.model_loader import (
    load_base_model_and_tokenizer,
    setup_lora,
    get_gpu_memory_info,
)
from training.data_loader import load_sft_dataset
from training.mlflow_utils import ExperimentTracker, ModelRegistry


def train_sft(tenant_id: str, config_override: dict = None):
    """
    Run SFT training for a specific tenant.

    Args:
        tenant_id: "sis" or "mfg"
        config_override: Optional config overrides
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"SFT TRAINING — TENANT: {tenant_id.upper()}")
    logger.info(f"{'='*70}")

    # Load config
    config = get_sft_config(tenant_id)
    if config_override:
        config.update(config_override)

    train_cfg = config["training"]
    model_cfg = config["model"]
    tenant_cfg = config["tenant"]

    # Initialize tracking
    tracker = ExperimentTracker(
        experiment_name=f"sft-{tenant_id}",
    )
    run_name = f"sft_{tenant_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    tracker.start_run(
        run_name=run_name,
        tags={
            "tenant_id": tenant_id,
            "domain": tenant_cfg["domain"],
            "model_type": "sft",
            "base_model": model_cfg["base_model"],
        },
    )

    try:
        # Log GPU info
        gpu_info = get_gpu_memory_info()
        logger.info(f"GPU: {gpu_info}")
        tracker.log_params({"gpu": gpu_info})

        # ---- Load model and tokenizer ----
        logger.info("Loading model and tokenizer...")
        model, tokenizer = load_base_model_and_tokenizer(config, for_training=True)

        # ---- Apply LoRA ----
        logger.info("Applying LoRA adapters...")
        model, peft_config = setup_lora(model, config)

        # Log config
        tracker.log_params({
            "base_model": model_cfg["base_model"],
            "max_seq_length": model_cfg["max_seq_length"],
            "lora_r": config["lora"]["r"],
            "lora_alpha": config["lora"]["lora_alpha"],
            "lora_dropout": config["lora"]["lora_dropout"],
            "learning_rate": train_cfg["learning_rate"],
            "num_epochs": train_cfg["num_train_epochs"],
            "batch_size": train_cfg["per_device_train_batch_size"],
            "grad_accum": train_cfg["gradient_accumulation_steps"],
            "effective_batch_size": (
                train_cfg["per_device_train_batch_size"]
                * train_cfg["gradient_accumulation_steps"]
            ),
        })

        # ---- Load dataset ----
        logger.info("Loading SFT dataset...")
        data_cfg = config["data"]
        train_dataset, eval_dataset = load_sft_dataset(
            train_path=data_cfg["train_path"],
            eval_path=data_cfg["eval_path"],
            tokenizer=tokenizer,
            max_seq_length=model_cfg["max_seq_length"],
        )

        tracker.log_params({
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
        })
        logger.info(f"Dataset: {len(train_dataset)} train, {len(eval_dataset)} eval")

        # ---- Configure trainer ----
        from transformers import TrainingArguments
        from trl import SFTTrainer

        output_dir = train_cfg["output_dir"]
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=train_cfg["num_train_epochs"],
            per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
            per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            learning_rate=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
            warmup_ratio=train_cfg["warmup_ratio"],
            lr_scheduler_type=train_cfg["lr_scheduler_type"],
            logging_steps=train_cfg["logging_steps"],
            eval_strategy=train_cfg.get("eval_strategy", "steps"),
            eval_steps=train_cfg.get("eval_steps", 20),
            save_strategy=train_cfg.get("save_strategy", "steps"),
            save_steps=train_cfg.get("save_steps", 50),
            save_total_limit=train_cfg.get("save_total_limit", 3),
            fp16=train_cfg.get("fp16", True),
            bf16=train_cfg.get("bf16", False),
            gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim=train_cfg.get("optim", "paged_adamw_8bit"),
            max_grad_norm=train_cfg.get("max_grad_norm", 1.0),
            seed=train_cfg.get("seed", 42),
            report_to="none",  # We handle tracking ourselves
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=False,
            dataloader_pin_memory=False,  # Save memory
        )

        # ---- Create trainer ----
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_seq_length=model_cfg["max_seq_length"],
            dataset_text_field="text",
            packing=False,
        )

        # ---- Train ----
        logger.info("Starting SFT training...")
        gpu_before = get_gpu_memory_info()
        logger.info(f"GPU before training: {gpu_before}")

        t_start = time.time()
        train_result = trainer.train()
        train_time = round(time.time() - t_start, 2)

        logger.info(f"Training completed in {train_time}s")

        # ---- Log training metrics ----
        train_metrics = train_result.metrics
        train_metrics["train_time_sec"] = train_time
        tracker.log_metrics({
            "train_loss": train_metrics.get("train_loss", 0),
            "train_runtime": train_metrics.get("train_runtime", 0),
            "train_samples_per_second": train_metrics.get("train_samples_per_second", 0),
            "train_time_sec": train_time,
        })

        # ---- Evaluate ----
        logger.info("Running evaluation...")
        eval_metrics = trainer.evaluate()
        tracker.log_metrics({
            "eval_loss": eval_metrics.get("eval_loss", 0),
            "eval_runtime": eval_metrics.get("eval_runtime", 0),
        })
        logger.info(f"Eval loss: {eval_metrics.get('eval_loss', 'N/A')}")

        # ---- Save adapter ----
        logger.info(f"Saving adapter to {output_dir}...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Save training metadata
        metadata = {
            "tenant_id": tenant_id,
            "model_type": "sft",
            "base_model": model_cfg["base_model"],
            "adapter_path": output_dir,
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "config": {
                "lora": config["lora"],
                "training": train_cfg,
            },
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
            model_type="sft",
            version=version,
            adapter_path=output_dir,
            base_model=model_cfg["base_model"],
            metrics={
                "train_loss": train_metrics.get("train_loss"),
                "eval_loss": eval_metrics.get("eval_loss"),
            },
            training_config=train_cfg,
            dataset_info={
                "train_samples": len(train_dataset),
                "eval_samples": len(eval_dataset),
            },
        )

        # Log model info
        tracker.log_model_info(
            tenant_id=tenant_id,
            model_type="sft",
            base_model=model_cfg["base_model"],
            adapter_path=output_dir,
            metrics=eval_metrics,
        )

        gpu_after = get_gpu_memory_info()
        logger.info(f"GPU after training: {gpu_after}")

        tracker.end_run(status="FINISHED")

        logger.info(f"\n{'='*70}")
        logger.info(f"SFT TRAINING COMPLETE — {tenant_id.upper()}")
        logger.info(f"Adapter saved: {output_dir}")
        logger.info(f"Train loss: {train_metrics.get('train_loss', 'N/A')}")
        logger.info(f"Eval loss: {eval_metrics.get('eval_loss', 'N/A')}")
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
        logger.error(f"Training failed: {e}")
        tracker.log_metrics({"error": 1})
        tracker.end_run(status="FAILED")
        raise
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="SFT Training")
    parser.add_argument(
        "--tenant", type=str, required=True,
        choices=["sis", "mfg"],
        help="Tenant to train for",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size",
    )
    args = parser.parse_args()

    overrides = {}
    if args.epochs:
        overrides.setdefault("training", {})["num_train_epochs"] = args.epochs
    if args.lr:
        overrides.setdefault("training", {})["learning_rate"] = args.lr
    if args.batch_size:
        overrides.setdefault("training", {})["per_device_train_batch_size"] = args.batch_size

    result = train_sft(args.tenant, config_override=overrides if overrides else None)
    print(f"\nResult: {json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    main()
