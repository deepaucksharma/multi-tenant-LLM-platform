"""
Model and tokenizer loading utilities.

Supports:
- 4-bit QLoRA when CUDA + bitsandbytes are available
- standard LoRA in fp16/fp32 when they are not
"""
import os
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from loguru import logger
from dotenv import load_dotenv

load_dotenv()


def resolve_device() -> str:
    """
    Resolve the training/inference device from env and torch runtime.

    DEVICE=auto|cuda|cpu is supported. ROCm builds still use "cuda" through torch.
    """
    requested = os.getenv("DEVICE", "auto").strip().lower()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("DEVICE=cuda requested but no CUDA/ROCm device is available; falling back to CPU")
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def is_rocm_build() -> bool:
    """Return True when torch is a ROCm build."""
    return getattr(torch.version, "hip", None) is not None


def can_use_bnb_4bit(config: Dict[str, Any]) -> bool:
    """Determine whether bitsandbytes 4-bit loading is actually usable."""
    quant_cfg = config.get("quantization", {})
    requested = str(os.getenv("USE_4BIT", "auto")).strip().lower()
    if requested == "false":
        return False
    if requested == "true":
        quant_requested = True
    else:
        quant_requested = quant_cfg.get("load_in_4bit", True)

    if not quant_requested:
        return False
    if resolve_device() != "cuda":
        return False
    if is_rocm_build():
        return False
    try:
        import bitsandbytes  # noqa: F401
        return True
    except ImportError:
        return False


def get_effective_torch_dtype(config: Dict[str, Any]):
    """Choose a dtype compatible with the active runtime."""
    dtype_str = config["model"].get("torch_dtype", "float16")
    preferred = getattr(torch, dtype_str)
    if resolve_device() == "cpu" and preferred == torch.float16:
        return torch.float32
    return preferred


def get_effective_optimizer(config: Dict[str, Any], prefer_bnb: bool = False) -> str:
    """
    Map bitsandbytes-specific optimizers to torch-native ones when needed.
    """
    requested = config.get("training", {}).get("optim", "paged_adamw_8bit")
    if prefer_bnb:
        return requested

    if requested.startswith("paged_adamw"):
        return "adamw_torch"
    return requested


def get_training_runtime_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return runtime-specific training argument overrides."""
    device = resolve_device()
    use_bnb = can_use_bnb_4bit(config)
    dtype = get_effective_torch_dtype(config)

    return {
        "device": device,
        "use_bnb_4bit": use_bnb,
        "torch_dtype": dtype,
        "optim": get_effective_optimizer(config, prefer_bnb=use_bnb),
        "fp16": device == "cuda" and dtype == torch.float16,
        "bf16": False,
    }


def load_base_model_and_tokenizer(
    config: Dict[str, Any],
    for_training: bool = True,
) -> Tuple:
    """
    Load the base model with 4-bit quantization and tokenizer.

    Args:
        config: Training config dict
        for_training: If True, prepares model for training

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = config["model"]
    quant_cfg = config.get("quantization", {})
    runtime = get_training_runtime_config(config)

    model_path = model_cfg.get("local_path", model_cfg["base_model"])
    if not Path(model_path).exists():
        model_path = model_cfg["base_model"]
        logger.info(f"Local model not found, using HF hub: {model_path}")

    logger.info(f"Loading model: {model_path}")
    torch_dtype = runtime["torch_dtype"]

    load_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "token": os.getenv("HF_TOKEN"),
        "attn_implementation": "eager",
    }

    if runtime["use_bnb_4bit"]:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_cfg.get("load_in_4bit", True),
            bnb_4bit_compute_dtype=getattr(
                torch,
                quant_cfg.get("bnb_4bit_compute_dtype", "float16"),
            ),
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        )
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = "auto"
        logger.info("Using bitsandbytes 4-bit loading")
    else:
        device = runtime["device"]
        if device == "cuda":
            load_kwargs["device_map"] = {"": "cuda:0"}
        else:
            load_kwargs["device_map"] = {"": "cpu"}
            load_kwargs["low_cpu_mem_usage"] = True
        logger.info(
            f"Using standard model loading on {device} "
            f"(dtype={str(torch_dtype).replace('torch.', '')})"
        )

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN"),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if for_training:
        tokenizer.padding_side = "right"
    else:
        tokenizer.padding_side = "left"

    if for_training:
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model loaded: {param_count/1e6:.1f}M params, "
        f"{trainable_count/1e6:.1f}M trainable"
    )

    return model, tokenizer


def setup_lora(model, config: Dict[str, Any]):
    """
    Apply LoRA adapters to the model.

    Args:
        model: The base model
        config: Training config dict containing 'lora' section

    Returns:
        (model_with_lora, peft_config)
    """
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

    lora_cfg = config["lora"]

    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    else:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"LoRA applied: {trainable_params/1e6:.2f}M trainable / "
        f"{total_params/1e6:.1f}M total "
        f"({100*trainable_params/total_params:.2f}%)"
    )

    model.print_trainable_parameters()

    return model, peft_config


def load_adapter(
    base_model,
    adapter_path: str,
    adapter_name: str = "default",
):
    """
    Load a pre-trained LoRA adapter onto the base model.

    Args:
        base_model: The base model
        adapter_path: Path to the adapter directory
        adapter_name: Name for the adapter

    Returns:
        Model with adapter loaded
    """
    from peft import PeftModel

    logger.info(f"Loading adapter from: {adapter_path}")

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        adapter_name=adapter_name,
        is_trainable=False,
    )

    logger.info(f"Adapter '{adapter_name}' loaded successfully")
    return model


def get_gpu_memory_info() -> Dict[str, float]:
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return {"available": False}

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_mem / 1024**3

    return {
        "available": True,
        "device": torch.cuda.get_device_name(0),
        "total_gb": round(total, 2),
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "free_gb": round(total - allocated, 2),
    }
