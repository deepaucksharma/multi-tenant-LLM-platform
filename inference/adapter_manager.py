"""
Adapter manager for multi-tenant model serving.
Handles loading base model once and swapping LoRA adapters per tenant.
"""
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, List
from threading import Condition, Lock

import torch
from loguru import logger
from dotenv import load_dotenv

from training.model_loader import get_gpu_memory_info, load_base_model_and_tokenizer

load_dotenv()


class AdapterManager:
    """
    Manages a single base model with multiple LoRA adapters.
    Thread-safe adapter swapping for multi-tenant inference.
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._base_loaded = False
        self._active_adapter: Optional[str] = None
        self._available_adapters: Dict[str, Dict] = {}
        self._lock = Lock()
        self._generation_state = Condition(self._lock)
        self._active_generations = 0
        self._load_count = 0
        self._generation_count = 0

    def load_base_model(self):
        """Load the base model and tokenizer using adaptive runtime settings.

        The runtime (device, 4-bit quantization) is resolved automatically from
        the environment via ``get_training_runtime_config()``. Set DEVICE and
        USE_4BIT env vars to override the auto-detected behaviour.
        """
        if self._base_loaded:
            logger.info("Base model already loaded")
            return

        local_path = "./models/base/qwen2.5-1.5b-instruct"
        # Build a minimal config understood by load_base_model_and_tokenizer.
        # The "training" key below is required only because get_training_runtime_config
        # reads config["training"]["optim"] to pick the optimizer; at inference time
        # the optimizer is never actually used — only the device/dtype/bnb flags matter.
        config = {
            "model": {
                "base_model": os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
                "local_path": local_path,
                "torch_dtype": "float16",
            },
            "quantization": {
                # load_in_4bit is advisory; can_use_bnb_4bit() will override it
                # based on the USE_4BIT env var and bitsandbytes availability.
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            },
            # Required by get_training_runtime_config for optimizer selection;
            # the optimizer itself is unused during inference.
            "training": {
                "optim": "paged_adamw_8bit",
            },
        }

        self._model, self._tokenizer = load_base_model_and_tokenizer(
            config,
            for_training=False,
        )
        self._model.eval()

        self._base_loaded = True
        self._scan_adapters()

        logger.info(f"Base model loaded. Available adapters: {list(self._available_adapters.keys())}")

    def _scan_adapters(self):
        """Scan for available adapter directories."""
        adapters_root = Path("models/adapters")
        if not adapters_root.exists():
            return

        for tenant_dir in adapters_root.iterdir():
            if not tenant_dir.is_dir():
                continue
            tenant_id = tenant_dir.name
            for adapter_dir in tenant_dir.iterdir():
                if not adapter_dir.is_dir():
                    continue
                adapter_type = adapter_dir.name
                config_path = adapter_dir / "adapter_config.json"
                if config_path.exists():
                    adapter_key = f"{tenant_id}_{adapter_type}"
                    self._available_adapters[adapter_key] = {
                        "tenant_id": tenant_id,
                        "adapter_type": adapter_type,
                        "path": str(adapter_dir),
                        "loaded": False,
                    }
                    logger.info(f"Found adapter: {adapter_key} at {adapter_dir}")

    def get_adapter_key(self, tenant_id: str, model_type: str = "sft") -> str:
        """Get the adapter key for a tenant and model type."""
        if model_type == "base":
            return ""

        key = f"{tenant_id}_{model_type}"
        if key not in self._available_adapters:
            # Fallback: try the other adapter-backed variants.
            for fallback_type in ["sft", "dpo"]:
                if fallback_type == model_type:
                    continue
                fallback_key = f"{tenant_id}_{fallback_type}"
                if fallback_key in self._available_adapters:
                    logger.info(f"Adapter {key} not found, falling back to {fallback_key}")
                    return fallback_key
            return ""  # No adapter available
        return key

    def _select_adapter_locked(self, adapter_key: Optional[str]) -> bool:
        """
        Select the adapter state that should be used for the next generation.

        When switching between adapters or between adapter/base modes, wait for
        any in-flight generations to finish so we never mutate model state under
        an active decode.
        """
        desired_adapter = adapter_key or None
        if desired_adapter == self._active_adapter:
            return True

        while self._active_generations > 0:
            self._generation_state.wait()

        if desired_adapter is None:
            self._active_adapter = None
            logger.info("Switched inference route to base model")
            return True

        if desired_adapter not in self._available_adapters:
            logger.warning(f"Adapter not found: {desired_adapter}")
            return False

        adapter_info = self._available_adapters[desired_adapter]
        adapter_path = adapter_info["path"]

        try:
            from peft import PeftModel

            if hasattr(self._model, "peft_config"):
                if desired_adapter not in self._model.peft_config:
                    self._model.load_adapter(adapter_path, adapter_name=desired_adapter)
                self._model.set_adapter(desired_adapter)
            else:
                self._model = PeftModel.from_pretrained(
                    self._model,
                    adapter_path,
                    adapter_name=desired_adapter,
                    is_trainable=False,
                )

            self._model.eval()
            self._active_adapter = desired_adapter
            adapter_info["loaded"] = True
            self._load_count += 1

            logger.info(f"Adapter hot-swapped: {desired_adapter}")
            return True

        except Exception as e:
            logger.error(
                f"Failed to hot-swap adapter '{desired_adapter}': {e}. "
                f"Active adapter remains: {self._active_adapter}"
            )
            return False

    def load_adapter(self, adapter_key: str) -> bool:
        """Load a specific adapter onto the base model via hot-swap."""
        if not self._base_loaded:
            self.load_base_model()

        if adapter_key not in self._available_adapters:
            logger.warning(f"Adapter not found: {adapter_key}")
            return False

        with self._lock:
            return self._select_adapter_locked(adapter_key)

    def use_base_model(self) -> bool:
        """Select base-model inference even if PEFT adapters are loaded in memory."""
        if not self._base_loaded:
            self.load_base_model()

        with self._lock:
            return self._select_adapter_locked(None)

    def generate(
        self,
        messages: List[Dict],
        adapter_key: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Generate a response from the current model state."""
        if not self._base_loaded:
            self.load_base_model()

        with self._lock:
            if not self._select_adapter_locked(adapter_key):
                raise RuntimeError(f"Unable to activate adapter '{adapter_key}'")

            try:
                prompt = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt = self._format_messages_fallback(messages)

            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048 - max_new_tokens,
                padding=True,
            ).to(self._model.device)
            model = self._model
            tokenizer = self._tokenizer
            use_base_model = adapter_key in ("", None) and hasattr(model, "disable_adapter")
            self._active_generations += 1

        try:
            adapter_context = model.disable_adapter() if use_base_model else nullcontext()
            with adapter_context:
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=max(temperature, 0.01),
                        top_p=top_p,
                        do_sample=do_sample,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

            input_length = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][input_length:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)

            with self._lock:
                self._generation_count += 1
            return response.strip()
        finally:
            with self._lock:
                self._active_generations -= 1
                self._generation_state.notify_all()

    def generate_stream(
        self,
        messages: List[Dict],
        adapter_key: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """Generate response tokens one at a time for streaming."""
        if not self._base_loaded:
            self.load_base_model()

        from transformers import TextIteratorStreamer
        from threading import Thread

        # Acquire lock only for tokenization + thread setup, then release
        # before yielding tokens so concurrent generate() calls are not blocked.
        with self._lock:
            if not self._select_adapter_locked(adapter_key):
                raise RuntimeError(f"Unable to activate adapter '{adapter_key}'")

            try:
                prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                prompt = self._format_messages_fallback(messages)

            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048 - max_new_tokens,
                padding=True,
            ).to(self._model.device)
            model = self._model
            tokenizer = self._tokenizer
            use_base_model = adapter_key in ("", None) and hasattr(model, "disable_adapter")

            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "temperature": max(temperature, 0.01),
                "top_p": top_p,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "streamer": streamer,
            }

            def _run_generation():
                adapter_context = model.disable_adapter() if use_base_model else nullcontext()
                with adapter_context:
                    with torch.inference_mode():
                        model.generate(**generation_kwargs)

            thread = Thread(target=_run_generation)
            thread.start()
            self._active_generations += 1
            self._generation_count += 1
        # Lock released here — token iteration happens without holding it

        try:
            for token in streamer:
                yield token
            thread.join()
        finally:
            with self._lock:
                self._active_generations -= 1
                self._generation_state.notify_all()


    def _format_messages_fallback(self, messages: List[Dict]) -> str:
        """Fallback message formatting."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    @property
    def is_loaded(self) -> bool:
        return self._base_loaded

    @property
    def active_adapter(self) -> Optional[str]:
        return self._active_adapter

    @property
    def available_adapters(self) -> Dict:
        return self._available_adapters

    def get_stats(self) -> Dict:
        local_path = Path("./models/base/qwen2.5-1.5b-instruct")
        local_model_ready = local_path.exists() and any(local_path.iterdir())
        configured_source = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        return {
            "base_loaded": self._base_loaded,
            "active_adapter": self._active_adapter,
            "available_adapters": list(self._available_adapters.keys()),
            "load_count": self._load_count,
            "generation_count": self._generation_count,
            "active_generations": self._active_generations,
            "gpu_memory": self._get_gpu_memory(),
            "ready": self._base_loaded or local_model_ready or bool(configured_source),
            "load_strategy": "lazy",
            "model_source": str(local_path) if local_model_ready else configured_source,
        }

    def _get_gpu_memory(self) -> Dict:
        return get_gpu_memory_info()


# Global singleton
_adapter_manager: Optional[AdapterManager] = None


def get_adapter_manager() -> AdapterManager:
    """Get the global adapter manager singleton."""
    global _adapter_manager
    if _adapter_manager is None:
        _adapter_manager = AdapterManager()
    return _adapter_manager
