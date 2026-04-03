"""
Merge LoRA adapter into base model and save as standalone model.
Useful for deployment or as a starting point for further training.

Usage:
    python training/merge_adapter.py --tenant sis --type sft
    python training/merge_adapter.py --tenant mfg --type dpo
"""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import torch
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


def merge_adapter(
    tenant_id: str,
    model_type: str = "sft",
    output_dir: str = None,
):
    """
    Merge a LoRA adapter into the base model.

    Args:
        tenant_id: Tenant identifier
        model_type: "sft" or "dpo"
        output_dir: Output directory for merged model
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    adapter_path = f"./models/adapters/{tenant_id}/{model_type}"
    if not Path(adapter_path).exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    if output_dir is None:
        output_dir = f"./models/merged/{tenant_id}_{model_type}"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load adapter config to get base model
    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
    else:
        base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    logger.info(f"Base model: {base_model_name}")
    logger.info(f"Adapter: {adapter_path}")
    logger.info(f"Output: {output_dir}")

    # Load base model in full precision for merging
    logger.info("Loading base model (full precision for merge)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",  # Use CPU for merging to save GPU memory
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN"),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN"),
    )

    # Load adapter
    logger.info("Loading adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    # Merge
    logger.info("Merging adapter into base model...")
    model = model.merge_and_unload()

    # Save
    logger.info(f"Saving merged model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save merge metadata
    merge_info = {
        "tenant_id": tenant_id,
        "model_type": model_type,
        "base_model": base_model_name,
        "adapter_path": adapter_path,
        "merged_at": datetime.utcnow().isoformat(),
    }
    (Path(output_dir) / "merge_info.json").write_text(
        json.dumps(merge_info, indent=2)
    )

    logger.info("Merge complete!")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter")
    parser.add_argument("--tenant", type=str, required=True, choices=["sis", "mfg"])
    parser.add_argument("--type", type=str, default="sft", choices=["sft", "dpo"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    merge_adapter(args.tenant, args.type, args.output)


if __name__ == "__main__":
    main()
