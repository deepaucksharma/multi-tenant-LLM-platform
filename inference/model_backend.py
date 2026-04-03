"""
Inference backend selection for local development and deployment.

Supports:
- Hugging Face + PEFT adapter swapping (existing path)
- Ollama over HTTP (preferred local AMD/WSL path)
- HF Serverless Inference API (zero-GPU remote path)
"""
import os
import json
from typing import Dict, Iterable, List, Optional

import httpx
from loguru import logger

from inference.adapter_manager import get_adapter_manager
from inference.hf_inference_backend import HFInferenceBackend


class HFBackend:
    """Existing local Hugging Face backend."""

    name = "huggingface"

    def __init__(self):
        self._manager = get_adapter_manager()

    def warmup(self):
        self._manager.load_base_model()

    def prepare_route(self, route):
        if not self._manager.is_loaded:
            self._manager.load_base_model()

    def generate(
        self,
        messages: List[Dict],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tenant_id: Optional[str] = None,
        model_type: Optional[str] = None,
        adapter_key: Optional[str] = None,
    ) -> str:
        return self._manager.generate(
            messages,
            adapter_key=adapter_key,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def generate_stream(
        self,
        messages: List[Dict],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tenant_id: Optional[str] = None,
        model_type: Optional[str] = None,
        adapter_key: Optional[str] = None,
    ):
        return self._manager.generate_stream(
            messages,
            adapter_key=adapter_key,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def get_model_label(self, tenant_id: Optional[str], model_type: Optional[str], route) -> str:
        return route.adapter_key or "base"

    def get_stats(self) -> Dict:
        stats = self._manager.get_stats()
        stats["backend"] = self.name
        return stats


class OllamaBackend:
    """Ollama HTTP backend for AMD-friendly local inference."""

    name = "ollama"

    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        self.timeout = float(os.getenv("OLLAMA_TIMEOUT_SEC", "120"))
        self._warned_routes = set()

    def warmup(self):
        self._assert_available()

    def prepare_route(self, route):
        resolved = self._resolve_model_name(route.tenant_id, route.model_type)
        if route.adapter_key:
            default_candidates = {
                os.getenv("OLLAMA_MODEL_DEFAULT"),
                os.getenv("OLLAMA_MODEL"),
                "qwen2.5:1.5b",
            }
            if resolved in default_candidates:
                warning_key = (route.tenant_id, route.model_type, route.adapter_key)
                if warning_key not in self._warned_routes:
                    logger.warning(
                        "Ollama backend is serving '{}' for {}/{} while local adapter '{}' exists. "
                        "Configure OLLAMA_MODEL_{}_{} or register a tenant-specific Ollama model "
                        "to avoid bypassing PEFT adapters.",
                        resolved,
                        route.tenant_id,
                        route.model_type,
                        route.adapter_key,
                        route.tenant_id.upper(),
                        route.model_type.upper(),
                    )
                    self._warned_routes.add(warning_key)
        return None

    def generate(
        self,
        messages: List[Dict],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tenant_id: Optional[str] = None,
        model_type: Optional[str] = None,
        adapter_key: Optional[str] = None,
    ) -> str:
        model = self._resolve_model_name(tenant_id, model_type)
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        return data.get("message", {}).get("content", "").strip()

    def generate_stream(
        self,
        messages: List[Dict],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        tenant_id: Optional[str] = None,
        model_type: Optional[str] = None,
        adapter_key: Optional[str] = None,
    ) -> Iterable[str]:
        model = self._resolve_model_name(tenant_id, model_type)
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        }
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError("Received malformed Ollama stream chunk") from exc
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content
        except Exception as exc:
            logger.error(f"Ollama streaming request failed: {exc}")
            raise RuntimeError("Ollama streaming request failed") from exc

    def get_stats(self) -> Dict:
        model_map = {}
        for tenant_id in ("sis", "mfg"):
            for model_type in ("base", "sft", "dpo"):
                model_map[f"{tenant_id}_{model_type}"] = self._resolve_model_name(
                    tenant_id,
                    model_type,
                )

        stats = {
            "backend": self.name,
            "base_loaded": self.is_available(),
            "ready": self.is_available(),
            "load_strategy": "remote",
            "active_adapter": None,
            "available_adapters": list(model_map.keys()),
            "load_count": 0,
            "generation_count": None,
            "gpu_memory": {"available": None},
            "ollama_base_url": self.base_url,
            "resolved_models": model_map,
            "available_models": self.list_models() if self.is_available() else [],
        }
        return stats

    def get_model_label(self, tenant_id: Optional[str], model_type: Optional[str], route) -> str:
        return self._resolve_model_name(tenant_id, model_type)

    def is_available(self) -> bool:
        try:
            self._assert_available()
            return True
        except Exception:
            return False

    def _assert_available(self):
        with httpx.Client(timeout=min(self.timeout, 5.0)) as client:
            response = client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()

    def list_models(self) -> List[str]:
        with httpx.Client(timeout=min(self.timeout, 5.0)) as client:
            response = client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            payload = response.json()
        return [model.get("name", "") for model in payload.get("models", []) if model.get("name")]

    def _resolve_model_name(self, tenant_id: Optional[str], model_type: Optional[str]) -> str:
        tenant = (tenant_id or "default").upper()
        model_variant = (model_type or "sft").upper()

        candidates = [
            f"OLLAMA_MODEL_{tenant}_{model_variant}",
            f"OLLAMA_MODEL_{tenant}",
            f"OLLAMA_MODEL_{model_variant}",
            "OLLAMA_MODEL_DEFAULT",
            "OLLAMA_MODEL",
        ]
        for env_name in candidates:
            value = os.getenv(env_name)
            if value:
                return value
        return "qwen2.5:1.5b"


_backend = None


def get_model_backend():
    """Select the configured inference backend once per process."""
    global _backend
    if _backend is not None:
        return _backend

    requested = os.getenv("INFERENCE_BACKEND", "auto").strip().lower()
    ollama = OllamaBackend()

    if requested == "ollama":
        _backend = ollama
    elif requested == "hf":
        _backend = HFBackend()
    elif requested == "hf_inference":
        _backend = HFInferenceBackend()
    elif requested == "auto":
        if ollama.is_available():
            _backend = ollama
        elif os.getenv("HF_TOKEN"):
            _backend = HFInferenceBackend()
        else:
            _backend = HFBackend()
    else:
        raise ValueError(
            "INFERENCE_BACKEND must be one of: auto, ollama, hf, hf_inference"
        )

    logger.info(f"Selected inference backend: {_backend.name}")
    return _backend


def reset_model_backend():
    """Reset backend singleton for tests or runtime reconfiguration."""
    global _backend
    _backend = None
