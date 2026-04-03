"""
Focused tests for the runtime-adaptive training and serving paths.
These avoid real model loading and validate the control flow around it.
"""
import asyncio
from types import SimpleNamespace


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
