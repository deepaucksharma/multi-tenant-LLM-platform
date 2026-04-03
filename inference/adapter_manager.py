"""
Adapter manager for multi-tenant model serving.
Handles loading base model once and swapping LoRA adapters per tenant.
"""
import os
import json
import time
from pathlib import Path
from typing import Dict, Optional, List
from threading import Lock

import torch
from loguru import logger
from dotenv import load_dotenv

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
        self._load_count = 0
        self._generation_count = 0

    def load_base_model(self):
        """Load the base model and tokenizer (4-bit quantized)."""
        if self._base_loaded:
            logger.info("Base model already loaded")
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_path = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        local_path = "./models/base/qwen2.5-1.5b-instruct"

        if Path(local_path).exists():
            actual_path = local_path
        else:
            actual_path = model_path

        logger.info(f"Loading base model: {actual_path}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            actual_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN"),
            attn_implementation="eager",
        )
        self._model.eval()

        self._tokenizer = AutoTokenizer.from_pretrained(
            actual_path,
            trust_remote_code=True,
            token=os.getenv("HF_TOKEN"),
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._tokenizer.padding_side = "left"

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
        key = f"{tenant_id}_{model_type}"
        if key not in self._available_adapters:
            # Fallback: try other types
            for fallback_type in ["sft", "dpo", "base"]:
                fallback_key = f"{tenant_id}_{fallback_type}"
                if fallback_key in self._available_adapters:
                    logger.info(f"Adapter {key} not found, falling back to {fallback_key}")
                    return fallback_key
            return ""  # No adapter available
        return key

    def load_adapter(self, adapter_key: str) -> bool:
        """Load a specific adapter onto the base model via hot-swap."""
        if not self._base_loaded:
            self.load_base_model()

        if adapter_key not in self._available_adapters:
            logger.warning(f"Adapter not found: {adapter_key}")
            return False

        with self._lock:
            if self._active_adapter == adapter_key:
                return True  # Already active — no-op

            adapter_info = self._available_adapters[adapter_key]
            adapter_path = adapter_info["path"]

            try:
                from peft import PeftModel

                if hasattr(self._model, 'peft_config'):
                    # Model is already a PeftModel — use hot-swap API.
                    # load_adapter registers the adapter weights under adapter_key;
                    # set_adapter makes it the active one. This preserves all other
                    # loaded adapters in memory without destroying the model object.
                    if adapter_key not in self._model.peft_config:
                        self._model.load_adapter(adapter_path, adapter_name=adapter_key)
                    self._model.set_adapter(adapter_key)
                else:
                    # First adapter load: wrap the base model with PEFT.
                    self._model = PeftModel.from_pretrained(
                        self._model,
                        adapter_path,
                        adapter_name=adapter_key,
                        is_trainable=False,
                    )

                self._model.eval()
                self._active_adapter = adapter_key
                adapter_info["loaded"] = True
                self._load_count += 1

                logger.info(f"Adapter hot-swapped: {adapter_key}")
                return True

            except Exception as e:
                # Log but do NOT overwrite self._model — that would destroy
                # any already-loaded adapters and break concurrent requests.
                logger.error(
                    f"Failed to hot-swap adapter '{adapter_key}': {e}. "
                    f"Active adapter remains: {self._active_adapter}"
                )
                return False

    def generate(
        self,
        messages: List[Dict],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Generate a response from the current model state."""
        if not self._base_loaded:
            self.load_base_model()

        with self._lock:
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

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=max(temperature, 0.01),
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )

            input_length = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][input_length:]
            response = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

            self._generation_count += 1
            return response.strip()

    def generate_stream(
        self,
        messages: List[Dict],
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

            streamer = TextIteratorStreamer(
                self._tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "temperature": max(temperature, 0.01),
                "top_p": top_p,
                "do_sample": True,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
                "streamer": streamer,
            }

            thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
            thread.start()
            self._generation_count += 1
        # Lock released here — token iteration happens without holding it

        for token in streamer:
            yield token

        thread.join()


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
        return {
            "base_loaded": self._base_loaded,
            "active_adapter": self._active_adapter,
            "available_adapters": list(self._available_adapters.keys()),
            "load_count": self._load_count,
            "generation_count": self._generation_count,
            "gpu_memory": self._get_gpu_memory(),
        }

    def _get_gpu_memory(self) -> Dict:
        if torch.cuda.is_available():
            return {
                "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
                "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
            }
        return {"available": False}


# Global singleton
_adapter_manager: Optional[AdapterManager] = None


def get_adapter_manager() -> AdapterManager:
    """Get the global adapter manager singleton."""
    global _adapter_manager
    if _adapter_manager is None:
        _adapter_manager = AdapterManager()
    return _adapter_manager
