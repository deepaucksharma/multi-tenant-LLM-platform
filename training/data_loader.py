"""
Dataset loading and formatting for SFT and DPO training.
Handles conversion from JSON to HuggingFace Datasets with proper tokenization.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

from datasets import Dataset
from loguru import logger


def load_sft_dataset(
    train_path: str,
    eval_path: str,
    tokenizer=None,
    max_seq_length: int = 512,
) -> tuple:
    """
    Load SFT dataset in chat format.

    Args:
        train_path: Path to training JSON
        eval_path: Path to evaluation JSON
        tokenizer: Tokenizer for formatting
        max_seq_length: Maximum sequence length

    Returns:
        (train_dataset, eval_dataset)
    """
    train_data = _load_json(train_path)
    eval_data = _load_json(eval_path)

    logger.info(f"Loaded SFT data: {len(train_data)} train, {len(eval_data)} eval")

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    if tokenizer is not None:
        def format_chat(example):
            messages = example["messages"]
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                text = _manual_chat_format(messages)
            return {"text": text}

        train_dataset = train_dataset.map(format_chat, remove_columns=train_dataset.column_names)
        eval_dataset = eval_dataset.map(format_chat, remove_columns=eval_dataset.column_names)

    logger.info(f"SFT datasets ready: train={len(train_dataset)}, eval={len(eval_dataset)}")
    return train_dataset, eval_dataset


def load_dpo_dataset(
    train_path: str,
    eval_path: str,
    tokenizer=None,
) -> tuple:
    """
    Load DPO preference dataset.

    Args:
        train_path: Path to training JSON (TRL format)
        eval_path: Path to evaluation JSON (TRL format)
        tokenizer: Tokenizer for formatting

    Returns:
        (train_dataset, eval_dataset)
    """
    train_data = _load_json(train_path)
    eval_data = _load_json(eval_path)

    logger.info(f"Loaded DPO data: {len(train_data)} train, {len(eval_data)} eval")

    def format_dpo(examples: List[Dict]) -> Dict:
        formatted = {
            "prompt": [],
            "chosen": [],
            "rejected": [],
        }

        for ex in examples:
            if isinstance(ex.get("prompt"), list):
                prompt_msgs = ex["prompt"]
                chosen_msgs = ex["chosen"]
                rejected_msgs = ex["rejected"]

                if tokenizer is not None:
                    try:
                        prompt_text = tokenizer.apply_chat_template(
                            prompt_msgs, tokenize=False, add_generation_prompt=True
                        )
                        chosen_text = chosen_msgs[-1]["content"] if chosen_msgs else ""
                        rejected_text = rejected_msgs[-1]["content"] if rejected_msgs else ""
                    except Exception:
                        prompt_text = _manual_chat_format(prompt_msgs, add_generation_prompt=True)
                        chosen_text = chosen_msgs[-1]["content"] if chosen_msgs else ""
                        rejected_text = rejected_msgs[-1]["content"] if rejected_msgs else ""
                else:
                    prompt_text = _manual_chat_format(prompt_msgs, add_generation_prompt=True)
                    chosen_text = chosen_msgs[-1]["content"] if chosen_msgs else ""
                    rejected_text = rejected_msgs[-1]["content"] if rejected_msgs else ""
            else:
                prompt_text = ex.get("prompt", "")
                chosen_text = ex.get("chosen", "")
                rejected_text = ex.get("rejected", "")

            formatted["prompt"].append(prompt_text)
            formatted["chosen"].append(chosen_text)
            formatted["rejected"].append(rejected_text)

        return formatted

    formatted_train = format_dpo(train_data)
    formatted_eval = format_dpo(eval_data)

    train_dataset = Dataset.from_dict(formatted_train)
    eval_dataset = Dataset.from_dict(formatted_eval)

    logger.info(f"DPO datasets ready: train={len(train_dataset)}, eval={len(eval_dataset)}")
    return train_dataset, eval_dataset


def _load_json(path: str) -> List[Dict]:
    """Load JSON file."""
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    with open(filepath) as f:
        data = json.load(f)

    return data


def _manual_chat_format(
    messages: List[Dict],
    add_generation_prompt: bool = False,
) -> str:
    """Fallback chat formatting when tokenizer template is unavailable."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"<|system|>\n{content}")
        elif role == "user":
            parts.append(f"<|user|>\n{content}")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}")

    text = "\n".join(parts)
    if add_generation_prompt:
        text += "\n<|assistant|>\n"

    return text
