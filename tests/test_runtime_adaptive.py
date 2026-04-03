"""
Focused tests for the runtime-adaptive training and serving paths.
These avoid real model loading and validate the control flow around it.
"""
import asyncio
from types import SimpleNamespace

import numpy as np


def test_get_gpu_memory_info_uses_total_memory(monkeypatch):
    from training import model_loader

    class FakeProps:
        total_memory = 8 * 1024**3

    # Force CUDA path — prevent early-return via DirectML when DEVICE=dml in .env
    monkeypatch.setenv("DEVICE", "cuda")
    monkeypatch.setattr(model_loader.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(model_loader.torch.cuda, "memory_allocated", lambda: 2 * 1024**3)
    monkeypatch.setattr(model_loader.torch.cuda, "memory_reserved", lambda: 3 * 1024**3)
    monkeypatch.setattr(model_loader.torch.cuda, "get_device_properties", lambda idx: FakeProps())
    monkeypatch.setattr(model_loader.torch.cuda, "get_device_name", lambda idx: "Fake GPU")

    stats = model_loader.get_gpu_memory_info()

    assert stats["available"] is True
    assert stats["total_gb"] == 8.0
    assert stats["allocated_gb"] == 2.0
    assert stats["reserved_gb"] == 3.0
    assert stats["free_gb"] == 6.0


def test_base_model_route_never_falls_back_to_adapter():
    from inference.adapter_manager import AdapterManager

    manager = AdapterManager()
    manager._available_adapters = {
        "sis_sft": {"path": "models/adapters/sis/sft"},
    }

    assert manager.get_adapter_key("sis", "base") == ""
    assert manager.get_adapter_key("sis", "dpo") == "sis_sft"


def test_hf_backend_prepare_and_generate_use_runtime_route(monkeypatch):
    from inference import model_backend

    class FakeManager:
        def __init__(self):
            self.is_loaded = False
            self.load_calls = 0
            self.last_generate = None

        def load_base_model(self):
            self.load_calls += 1
            self.is_loaded = True

        def generate(self, messages, adapter_key=None, **kwargs):
            self.last_generate = {
                "messages": messages,
                "adapter_key": adapter_key,
                "kwargs": kwargs,
            }
            return "ok"

        def generate_stream(self, messages, adapter_key=None, **kwargs):
            self.last_generate = {
                "messages": messages,
                "adapter_key": adapter_key,
                "kwargs": kwargs,
            }
            yield "ok"

        def get_stats(self):
            return {"base_loaded": self.is_loaded}

    fake_manager = FakeManager()
    monkeypatch.setattr(model_backend, "get_adapter_manager", lambda: fake_manager)

    backend = model_backend.HFBackend()
    route = SimpleNamespace(adapter_key="sis_sft", tenant_id="sis", model_type="sft")

    backend.prepare_route(route)
    result = backend.generate([{"role": "user", "content": "hello"}], adapter_key=route.adapter_key)

    assert fake_manager.load_calls == 1
    assert result == "ok"
    assert fake_manager.last_generate["adapter_key"] == "sis_sft"


def test_health_reports_ready_for_lazy_hf_backend(monkeypatch):
    from inference import app as inference_app

    class FakeBackend:
        def get_stats(self):
            return {
                "backend": "huggingface",
                "base_loaded": False,
                "ready": True,
                "load_strategy": "lazy",
                "gpu_memory": {"available": False},
                "available_adapters": [],
            }

    monkeypatch.setattr(inference_app, "get_model_backend", lambda: FakeBackend())

    response = asyncio.run(inference_app.health_check())

    assert response.status == "ready"


def test_resolve_device_falls_back_to_cpu_when_directml_probe_crashes(monkeypatch):
    import sys
    from types import SimpleNamespace
    from training import model_loader

    monkeypatch.setenv("DEVICE", "auto")
    monkeypatch.setattr(model_loader.torch.cuda, "is_available", lambda: False)

    fake_directml = SimpleNamespace(
        is_available=lambda: (_ for _ in ()).throw(RuntimeError("dml boom"))
    )
    monkeypatch.setitem(sys.modules, "torch_directml", fake_directml)

    assert model_loader.resolve_device() == "cpu"


def test_embedding_fallback_returns_deterministic_vectors(monkeypatch):
    from rag import embeddings

    embeddings._embed_model = None

    def fail_import(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        fail_import,
        raising=True,
    )

    first = embeddings.embed_query("enrollment process")
    second = embeddings.embed_query("enrollment process")
    batch = embeddings.embed_texts(["enrollment process", "safety check"])

    assert len(first) == embeddings.get_embedding_dimension()
    assert first == second
    assert isinstance(batch, np.ndarray)
    assert batch.shape[0] == 2


def test_smoke_model_source_auto_creates_local_tiny_model(monkeypatch, tmp_path):
    from training import model_loader

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SMOKE_TEST_LOCAL_MODEL_PATH", "")
    monkeypatch.setenv("SMOKE_TEST_BASE_MODEL", "")

    data_dir = tmp_path / "data" / "sis" / "sft"
    data_dir.mkdir(parents=True)
    sample = [
        {
            "messages": [
                {"role": "user", "content": "What is enrollment?"},
                {"role": "assistant", "content": "Enrollment requires documents."},
            ]
        }
    ]
    (data_dir / "train_chat.json").write_text(__import__("json").dumps(sample))
    (data_dir / "eval_chat.json").write_text(__import__("json").dumps(sample))

    config = {
        "model": {"base_model": "remote/model", "local_path": "./missing", "max_seq_length": 128},
        "data": {
            "train_path": str(data_dir / "train_chat.json"),
            "eval_path": str(data_dir / "eval_chat.json"),
        },
        "smoke_test": {"enabled": True},
    }

    path = model_loader._ensure_local_smoke_model(config)

    assert (tmp_path / "models" / "base" / "smoke-gpt2" / "config.json").exists()
    assert path.endswith("models/base/smoke-gpt2")
