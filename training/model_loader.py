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
import json

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Directories that are allowed as roots for local model paths.
# Only paths that resolve to within one of these roots are accepted.
_ALLOWED_MODEL_ROOTS: Tuple[Path, ...] = (
    Path("./models").resolve(),
    Path(".").resolve(),
)

def _safe_model_path(path_str: str) -> Optional[str]:
    """
    Resolve *path_str* and confirm it lives under one of the allowed model
    roots.  Returns the original string if it passes, or ``None`` if the
    resolved path would escape the allowed roots (directory traversal guard).

    Remote identifiers (no '/' and not starting with '.') are passed through
    unchanged because they are Hub model names such as
    ``"Qwen/Qwen2.5-1.5B-Instruct"`` and must not be treated as file paths.
    """
    # Heuristic: Hub IDs look like "org/model" or "model" — they are not
    # filesystem paths.  Only validate strings that look like paths.
    if not path_str or ("/" not in path_str and not path_str.startswith(".")):
        return path_str  # Hub identifier — no filesystem check needed

    resolved = Path(path_str).resolve()
    for root in _ALLOWED_MODEL_ROOTS:
        try:
            resolved.relative_to(root)
            return path_str  # ✓ within an allowed root
        except ValueError:
            continue
    logger.error(
        f"Path traversal attempt blocked: '{path_str}' resolves to '{resolved}' "
        f"which is outside allowed model roots {[str(r) for r in _ALLOWED_MODEL_ROOTS]}"
    )
    return None


def resolve_device() -> str:
    """
    Resolve the training/inference device from env and torch runtime.

    DEVICE=auto|cuda|cpu|dml is supported. ROCm builds still use "cuda" through torch.
    """
    requested = os.getenv("DEVICE", "auto").strip().lower()
    if requested == "cpu":
        return "cpu"
    if requested == "dml":
        try:
            import torch_directml
            if torch_directml.is_available():
                return "dml"
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"DEVICE=dml requested but DirectML probe failed: {exc}; falling back to CPU")
        logger.warning("DEVICE=dml requested but torch-directml is not available; falling back to CPU")
        return "cpu"
    if requested == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("DEVICE=cuda requested but no CUDA/ROCm device is available; falling back to CPU")
        return "cpu"
        
    try:
        import torch_directml
        if torch_directml.is_available():
            return "dml"
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"DirectML probe failed during auto device detection: {exc}; ignoring DirectML")
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
    if resolve_device() in ("cpu", "dml") and preferred == torch.float16:
        # float16 is unsafe for CPU training and DML training (instability)
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
        "fp16": device in ("cuda", "dml") and dtype == torch.float16,
        "bf16": False,
    }


def resolve_model_source(config: Dict[str, Any], smoke_test: bool = False) -> str:
    """
    Resolve the model source with support for dedicated smoke-test overrides.

    Priority:
    1. smoke-test env overrides
    2. config local path if it exists
    3. smoke-test remote override
    4. config base model

    All filesystem paths are validated by ``_safe_model_path`` to prevent
    directory traversal (CWE-22).  Any path that resolves outside the allowed
    model roots is blocked and skipped.
    """
    model_cfg = config["model"]

    if smoke_test:
        smoke_local = _safe_model_path(os.getenv("SMOKE_TEST_LOCAL_MODEL_PATH", "").strip())
        if smoke_local and Path(smoke_local).exists():
            return smoke_local

    model_path = _safe_model_path(model_cfg.get("local_path", model_cfg["base_model"]))
    if model_path and Path(model_path).exists():
        return model_path

    if smoke_test:
        smoke_remote = os.getenv("SMOKE_TEST_BASE_MODEL", "").strip()
        if smoke_remote:
            return smoke_remote

    return model_cfg["base_model"]


def _ensure_local_smoke_model(config: Dict[str, Any]) -> str:
    """
    Build a tiny local GPT-2 style model + tokenizer for offline smoke tests.

    This keeps the full training code path exercisable on constrained systems
    without requiring a network download.
    """
    smoke_dir = Path("./models/base/smoke-gpt2")
    config_path = smoke_dir / "config.json"
    if config_path.exists():
        return str(smoke_dir)

    logger.info(f"Creating local smoke-test model at {smoke_dir}")
    smoke_dir.mkdir(parents=True, exist_ok=True)

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    data_paths = [
        Path(config["data"]["train_path"]),
        Path(config["data"]["eval_path"]),
    ]
    corpus: list[str] = []
    for path in data_paths:
        if not path.exists():
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
            for row in rows[:64]:
                for msg in row.get("messages", []):
                    content = msg.get("content", "").strip()
                    if content:
                        corpus.append(content)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not read smoke corpus from {path}: {exc}")

    if not corpus:
        corpus = [
            "Students must provide proof of residency and a birth certificate.",
            "Operators must follow lockout tagout before maintenance.",
            "Attendance must be recorded within fifteen minutes of class start.",
            "Stop production and notify quality if a class one defect is found.",
        ]

    chat_special_tokens = ["<|system|>", "<|user|>", "<|assistant|>"]
    tokenizer_backend = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"] + chat_special_tokens,
        min_frequency=1,
    )
    tokenizer_backend.train_from_iterator(corpus, trainer=trainer)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}<|system|>\n{% elif message['role'] == 'user' %}<|user|>\n{% elif message['role'] == 'assistant' %}<|assistant|>\n{% endif %}{{ message['content'] }}\n{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
    smoke_seq_len = int(os.getenv("SMOKE_SEQ_LEN", "64"))
    tokenizer.model_max_length = min(smoke_seq_len, int(config["model"].get("max_seq_length", 256)))
    tokenizer.save_pretrained(smoke_dir)

    smoke_seq_len = int(os.getenv("SMOKE_SEQ_LEN", "64"))
    smoke_n_layer = int(os.getenv("SMOKE_N_LAYER", "1"))
    smoke_n_embd = int(os.getenv("SMOKE_N_EMBD", "32"))
    smoke_n_head = max(1, smoke_n_embd // 64)
    smoke_n_positions = int(os.getenv("SMOKE_N_POSITIONS", "256"))

    model_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=smoke_n_positions,
        n_ctx=smoke_n_positions,
        n_embd=smoke_n_embd,
        n_layer=smoke_n_layer,
        n_head=smoke_n_head,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(model_config)
    model.save_pretrained(smoke_dir)
    logger.info(
        f"Smoke model: vocab={tokenizer.vocab_size}, "
        f"n_layer={smoke_n_layer}, n_embd={smoke_n_embd}, "
        f"n_head={smoke_n_head}, n_ctx={smoke_n_positions}, "
        f"tokenizer_max_len={smoke_seq_len}, "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )
    return str(smoke_dir)


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
    smoke_test = config.get("smoke_test", {}).get("enabled", False)
    model_path = resolve_model_source(config, smoke_test=smoke_test)
    if smoke_test and not Path(model_path).exists():
        model_path = _ensure_local_smoke_model(config)
    if not Path(model_path).exists():
        logger.info(f"Local model not found, using model source: {model_path}")

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
        elif device == "dml":
            # DirectML does not support multi-threaded .to(device) during HF weight loading.
            # Load to CPU first, then move to DML after model is fully constructed.
            load_kwargs["device_map"] = {"": "cpu"}
            load_kwargs["low_cpu_mem_usage"] = True
        else:
            load_kwargs["device_map"] = {"": "cpu"}
            load_kwargs["low_cpu_mem_usage"] = True
        logger.info(
            f"Using standard model loading on {device} "
            f"(dtype={str(torch_dtype).replace('torch.', '')})"
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    except Exception as exc:
        if smoke_test:
            raise RuntimeError(
                "Smoke-test model could not be loaded. "
                "Set SMOKE_TEST_LOCAL_MODEL_PATH to a local tiny model directory "
                "or set SMOKE_TEST_BASE_MODEL to a downloadable small model."
            ) from exc
        raise

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

    # For DirectML: move model to GPU after full construction on CPU
    device = runtime["device"]
    if device == "dml":
        import torch_directml
        dml_device = torch_directml.device()
        logger.info(f"Moving model to DirectML device: {dml_device}")
        model = model.to(dml_device)

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

    target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj"])
    available_module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    if not any(module in available_module_names for module in target_modules):
        fallback_targets = {
            "gpt2": ["c_attn", "c_proj"],
            "gpt_neo": ["q_proj", "v_proj"],
            "llama": ["q_proj", "k_proj", "v_proj", "o_proj"],
        }
        model_type = getattr(getattr(model, "config", None), "model_type", "")
        if model_type in fallback_targets:
            logger.warning(
                f"Configured LoRA target modules {target_modules} were not found for "
                f"model_type={model_type}; using {fallback_targets[model_type]} instead."
            )
            target_modules = fallback_targets[model_type]

    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=target_modules,
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
    try:
        import torch_directml
        if os.getenv("DEVICE", "auto").strip().lower() == "dml" and torch_directml.is_available():
            return {
                "available": True, 
                "device": "DirectML Adapter",
                "total_gb": 0.0,
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "free_gb": 0.0,
            }
    except ImportError:
        pass

    if not torch.cuda.is_available():
        return {"available": False}

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    return {
        "available": True,
        "device": torch.cuda.get_device_name(0),
        "total_gb": round(total, 2),
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "free_gb": round(total - allocated, 2),
    }
