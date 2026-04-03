"""
HuggingFace Serverless Inference API backend.

Routes generation requests to HF's free Serverless Inference API over HTTPS,
using the OpenAI-compatible /v1/chat/completions endpoint.  No local GPU needed.

Model resolution order (most specific → least specific):
    HF_INFERENCE_MODEL_{TENANT}_{TYPE}
    HF_INFERENCE_MODEL_{TENANT}
    HF_INFERENCE_MODEL
    "Qwen/Qwen2.5-1.5B-Instruct"  (hard-coded fallback)

Rate-limit / model-loading responses (HTTP 429 / 503) are retried with
exponential back-off up to MAX_RETRIES times.
"""
import json
import os
import time
from typing import Dict, Iterable, List, Optional

import httpx
from loguru import logger


_HF_API_BASE = "https://api-inference.huggingface.co/models"
_DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
_MAX_RETRIES = 3
_RETRY_STATUSES = {429, 503}


class HFInferenceBackend:
    """HuggingFace Serverless Inference API backend (no local GPU required)."""

    name = "hf_inference"

    def __init__(self):
        self._token: Optional[str] = os.getenv("HF_TOKEN")
        self._timeout = float(os.getenv("HF_INFERENCE_TIMEOUT_SEC", "60"))

    # ── Interface methods ──────────────────────────────────────────────────────

    def warmup(self):
        """No-op — the API endpoint is always available (subject to rate limits)."""
        logger.info("HFInferenceBackend ready (serverless — no local warmup needed)")

    def prepare_route(self, route):
        """No per-route preparation needed for the remote API."""
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
        model = self._resolve_model(tenant_id, model_type)
        payload = self._build_payload(messages, max_new_tokens, temperature, top_p, stream=False)
        url = f"{_HF_API_BASE}/{model}/v1/chat/completions"

        response_data = self._post_with_retry(url, payload)
        try:
            return response_data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Unexpected HF Inference API response format: {response_data}") from exc

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
        model = self._resolve_model(tenant_id, model_type)
        payload = self._build_payload(messages, max_new_tokens, temperature, top_p, stream=True)
        url = f"{_HF_API_BASE}/{model}/v1/chat/completions"

        headers = self._auth_headers()
        try:
            with httpx.stream(
                "POST",
                url,
                json=payload,
                headers=headers,
                timeout=self._timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[len("data:"):].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        chunk = json.loads(raw)
                        content = chunk["choices"][0]["delta"].get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"HF Inference API streaming failed (HTTP {exc.response.status_code}): {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"HF Inference API streaming error: {exc}") from exc

    def get_model_label(self, tenant_id: Optional[str], model_type: Optional[str], route) -> str:
        return self._resolve_model(tenant_id, model_type)

    def get_stats(self) -> Dict:
        model_map = {}
        for tenant_id in ("sis", "mfg"):
            for model_type in ("base", "sft", "dpo"):
                model_map[f"{tenant_id}_{model_type}"] = self._resolve_model(tenant_id, model_type)

        return {
            "backend": self.name,
            "base_loaded": True,
            "ready": True,
            "load_strategy": "remote_api",
            "active_adapter": None,
            "available_adapters": list(model_map.keys()),
            "load_count": 0,
            "generation_count": None,
            "gpu_memory": {"available": None},
            "hf_inference_api_base": _HF_API_BASE,
            "resolved_models": model_map,
            "token_configured": bool(self._token),
        }

    # ── Internals ──────────────────────────────────────────────────────────────

    def _resolve_model(self, tenant_id: Optional[str], model_type: Optional[str]) -> str:
        tenant = (tenant_id or "default").upper()
        variant = (model_type or "sft").upper()

        candidates = [
            f"HF_INFERENCE_MODEL_{tenant}_{variant}",
            f"HF_INFERENCE_MODEL_{tenant}",
            "HF_INFERENCE_MODEL",
        ]
        for env_name in candidates:
            value = os.getenv(env_name)
            if value:
                return value
        return _DEFAULT_MODEL

    def _auth_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    @staticmethod
    def _build_payload(
        messages: List[Dict],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool,
    ) -> Dict:
        return {
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }

    def _post_with_retry(self, url: str, payload: Dict) -> Dict:
        headers = self._auth_headers()
        last_exc: Optional[Exception] = None

        for attempt in range(_MAX_RETRIES):
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.post(url, json=payload, headers=headers)

                if response.status_code not in _RETRY_STATUSES:
                    response.raise_for_status()
                    return response.json()

                # Retryable status
                wait = 2 ** attempt
                logger.warning(
                    "HF Inference API returned HTTP {} (attempt {}/{}); retrying in {}s…",
                    response.status_code,
                    attempt + 1,
                    _MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)
                last_exc = httpx.HTTPStatusError(
                    f"HTTP {response.status_code}", request=response.request, response=response
                )

            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in _RETRY_STATUSES:
                    raise RuntimeError(
                        f"HF Inference API error (HTTP {exc.response.status_code}): {exc}"
                    ) from exc
                wait = 2 ** attempt
                logger.warning(
                    "HF Inference API HTTP {} on attempt {}/{}; retrying in {}s…",
                    exc.response.status_code,
                    attempt + 1,
                    _MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)
                last_exc = exc

            except Exception as exc:
                raise RuntimeError(f"HF Inference API request failed: {exc}") from exc

        raise RuntimeError(
            f"HF Inference API unavailable after {_MAX_RETRIES} retries"
        ) from last_exc
