"""
Simplified DPO training that avoids loading two full models simultaneously.
Better for 8GB VRAM — uses implicit reference (no ref model copy).

Usage:
    python training/dpo_train_simple.py --tenant sis
    python training/dpo_train_simple.py --tenant mfg
"""
import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

from training.config_loader import get_dpo_config, get_sft_config
from training.model_loader import (
    load_base_model_and_tokenizer,
    setup_lora,
    get_gpu_memory_info,
)
from training.data_loader import load_dpo_dataset
from training.mlflow_utils import ExperimentTracker, ModelRegistry


def train_dpo_simple(tenant_id: str):
    """
    Simplified DPO: uses model itself as implicit reference via PEFT.
    TRL's DPOTrainer can use ref_model=None when the model is a PeftModel,
    in which case it uses the base (frozen) weights as the reference.
    This halves memory usage.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"DPO TRAINING (SIMPLE) — TENANT: {tenant_id.upper()}")
    logger.info(f"{'='*70}")

    # Load configs
    dpo_config = get_dpo_config()
    sft_config = get_sft_config(tenant_id)
    config = {**sft_config, "dpo": dpo_config.get("dpo", {}), "training": dpo_config.get("training", {})}

    train_cfg = config["training"]
    model_cfg = config["model"]
    dpo_cfg = config["dpo"]
    output_dir = f"./models/adapters/{tenant_id}/dpo"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Check for SFT adapter
    sft_adapter_path = f"./models/adapters/{tenant_id}/sft"
    has_sft = Path(sft_adapter_path).exists() and (Path(sft_adapter_path) / "adapter_config.json").exists()

    # Tracking
    tracker = ExperimentTracker(experiment_name=f"dpo-simple-{tenant_id}")
    run_name = f"dpo_simple_{tenant_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    tracker.start_run(run_name=run_name, tags={"tenant_id": tenant_id, "model_type": "dpo_simple"})

    try:
        gpu_info = get_gpu_memory_info()
        logger.info(f"GPU: {gpu_info}")

        # ---- Load base model ----
        logger.info("Loading base model...")
        model, tokenizer = load_base_model_and_tokenizer(config, for_training=True)

        # ---- If SFT adapter exists, merge it first ----
        if has_sft:
            logger.info(f"Merging SFT adapter from {sft_adapter_path}...")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, sft_adapter_path, is_trainable=True)
            model = model.merge_and_unload()
            logger.info("SFT adapter merged")

        # ---- Apply fresh LoRA for DPO ----
        dpo_lora_config = config["lora"].copy()
        dpo_lora_config["r"] = 8  # Smaller for DPO
        dpo_lora_config["lora_alpha"] = 16
        config["lora"] = dpo_lora_config

        model, peft_config = setup_lora(model, config)

        tracker.log_params({
            "dpo_beta": dpo_cfg.get("beta", 0.1),
            "dpo_lora_r": 8,
            "has_sft_base": has_sft,
            "learning_rate": train_cfg["learning_rate"],
            "num_epochs": train_cfg["num_train_epochs"],
        })

        # ---- Load DPO data ----
        train_dataset, eval_dataset = load_dpo_dataset(
            train_path=f"./data/{tenant_id}/dpo/train_trl.json",
            eval_path=f"./data/{tenant_id}/dpo/eval_trl.json",
            tokenizer=tokenizer,
        )
        logger.info(f"DPO data: {len(train_dataset)} train, {len(eval_dataset)} eval")

        # ---- Train with ref_model=None (implicit reference from PEFT base) ----
        from trl import DPOTrainer, DPOConfig

        dpo_training_config = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=train_cfg["num_train_epochs"],
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
            learning_rate=train_cfg["learning_rate"],
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            logging_steps=2,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            fp16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            seed=42,
            report_to="none",
            remove_unused_columns=False,
            beta=dpo_cfg.get("beta", 0.1),
            loss_type=dpo_cfg.get("loss_type", "sigmoid"),
            max_prompt_length=dpo_cfg.get("max_prompt_length", 256),
            max_length=dpo_cfg.get("max_length", 512),
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=None,  # Uses PEFT base weights as implicit reference
            args=dpo_training_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )

        # Train
        logger.info("Starting DPO training (simple mode, no explicit ref model)...")
        t_start = time.time()
        train_result = trainer.train()
        train_time = round(time.time() - t_start, 2)

        train_metrics = train_result.metrics
        eval_metrics = trainer.evaluate()

        tracker.log_metrics({
            "dpo_train_loss": train_metrics.get("train_loss", 0),
            "dpo_eval_loss": eval_metrics.get("eval_loss", 0),
            "train_time_sec": train_time,
        })

        # Save
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        metadata = {
            "tenant_id": tenant_id,
            "model_type": "dpo_simple",
            "base_model": model_cfg["base_model"],
            "sft_adapter": sft_adapter_path if has_sft else None,
            "adapter_path": output_dir,
            "dpo_config": dpo_cfg,
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "trained_at": datetime.utcnow().isoformat(),
            "train_time_sec": train_time,
        }
        (Path(output_dir) / "training_metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str)
        )

        # Register
        registry = ModelRegistry()
        version = f"v{datetime.utcnow().strftime('%Y%m%d%H%M')}"
        registry.register_model(
            tenant_id=tenant_id,
            model_type="dpo",
            version=version,
            adapter_path=output_dir,
            base_model=model_cfg["base_model"],
            metrics={"train_loss": train_metrics.get("train_loss"), "eval_loss": eval_metrics.get("eval_loss")},
        )

        tracker.end_run(status="FINISHED")

        logger.info(f"\nDPO COMPLETE — {tenant_id.upper()}")
        logger.info(f"Train loss: {train_metrics.get('train_loss')}")
        logger.info(f"Eval loss: {eval_metrics.get('eval_loss')}")
        logger.info(f"Time: {train_time}s")

        return {"status": "success", "tenant_id": tenant_id, "adapter_path": output_dir}

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", type=str, required=True, choices=["sis", "mfg"])
    args = parser.parse_args()
    train_dpo_simple(args.tenant)


if __name__ == "__main__":
    main()
