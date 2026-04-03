import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_pii_runtime_status_is_degraded_for_sis_without_presidio(monkeypatch):
    from tenant_data_pipeline import pii_redact

    monkeypatch.setattr(pii_redact, "PRESIDIO_AVAILABLE", False)

    status = pii_redact.get_pii_runtime_status("sis")

    assert status["compliance_status"] == "degraded"
    assert status["presidio_available"] is False
    assert "regex" in status["detection_engines"]
    assert status["warnings"]


def test_pii_overlap_dedup_prevents_double_redaction():
    from tenant_data_pipeline.pii_redact import PIIFinding, _deduplicate_findings, redact_text

    text = "Contact Alice Example at alice@example.com today."
    findings = [
        PIIFinding(
            doc_id="doc1",
            tenant_id="sis",
            pii_type="EMAIL",
            original="alice@example.com",
            redacted="[EMAIL_REDACTED]",
            start=25,
            end=42,
            confidence=0.85,
        ),
        PIIFinding(
            doc_id="doc1",
            tenant_id="sis",
            pii_type="PERSON",
            original="alice@example.com",
            redacted="[NAME_REDACTED]",
            start=25,
            end=42,
            confidence=0.95,
        ),
    ]

    deduped = _deduplicate_findings(findings)
    redacted = redact_text(text, deduped)

    assert len(deduped) == 1
    assert redacted.count("[") == 1
    assert "[NAME_REDACTED]" in redacted


def test_pipeline_failure_writes_partial_report(monkeypatch, tmp_path):
    from tenant_data_pipeline import run_pipeline

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_pipeline, "save_synthetic_documents", lambda: {"sis": 10, "mfg": 10})
    monkeypatch.setattr(run_pipeline, "ingest_all_tenants", lambda: (_ for _ in ()).throw(ValueError("boom")))

    with pytest.raises(RuntimeError) as exc:
        run_pipeline.run_full_pipeline()

    assert "2_ingestion" in str(exc.value)

    reports = sorted((tmp_path / "evaluation" / "reports").glob("pipeline_run_*.json"))
    assert reports, "expected a partial pipeline report to be written"

    report = json.loads(reports[-1].read_text())
    assert report["status"] == "failed"
    assert report["failed_stage"] == "2_ingestion"
    assert report["stages"]["1_synthetic_generation"]["status"] == "success"
    assert report["stages"]["2_ingestion"]["status"] == "failed"
    assert "traceback" in report["stages"]["2_ingestion"]


class _FakeCollection:
    def __init__(self, client, name, fail_on_add=False):
        self.client = client
        self.name = name
        self.fail_on_add = fail_on_add
        self.items = []
        self.metadata = {}

    def count(self):
        return len(self.items)

    def add(self, ids, documents, embeddings, metadatas):
        if self.fail_on_add:
            raise ValueError("bad chunk")
        self.items.extend(ids)

    def modify(self, name):
        del self.client.collections[self.name]
        self.name = name
        self.client.collections[name] = self


class _FakeClient:
    def __init__(self, fail_on_staging_add=False):
        self.collections = {}
        self.fail_on_staging_add = fail_on_staging_add

    def get_collection(self, name):
        if name not in self.collections:
            raise KeyError(name)
        return self.collections[name]

    def get_or_create_collection(self, name, metadata=None):
        if name not in self.collections:
            fail = self.fail_on_staging_add and "staging" in name
            self.collections[name] = _FakeCollection(self, name, fail_on_add=fail)
        self.collections[name].metadata = metadata or {}
        return self.collections[name]

    def delete_collection(self, name):
        if name not in self.collections:
            raise KeyError(name)
        del self.collections[name]

    def list_collections(self):
        return list(self.collections.values())


def test_build_tenant_index_swaps_staging_collection(monkeypatch, tmp_path):
    from rag import build_index

    tenant_dir = tmp_path / "sis_chunks"
    tenant_dir.mkdir()
    (tenant_dir / "chunks.json").write_text(json.dumps([
        {
            "content": "Enrollment requires proof of residency.",
            "chunk_id": "chunk-1",
            "tenant_id": "sis",
            "doc_id": "doc-1",
            "title": "Enrollment Policy",
            "topic": "enrollment",
            "chunk_index": 0,
            "total_chunks": 1,
            "source_file": "policy.txt",
            "char_count": 39,
            "word_count": 5,
        }
    ]))

    client = _FakeClient()
    monkeypatch.setattr(
        build_index,
        "TENANTS",
        {"sis": SimpleNamespace(chunks_dir=tenant_dir, domain="education")},
    )
    monkeypatch.setattr(build_index, "get_chroma_client", lambda: client)
    monkeypatch.setattr(build_index.time, "time", lambda: 1234.567)

    def fake_embed_texts(texts, show_progress=False):
        return np.ones((len(texts), 3))

    monkeypatch.setattr("rag.embeddings.embed_texts", fake_embed_texts)

    result = build_index.build_tenant_index("sis")

    assert result["status"] == "success"
    assert "tenant_sis_docs" in client.collections
    assert all("staging" not in name for name in client.collections)


def test_build_tenant_index_cleans_up_failed_staging_collection(monkeypatch, tmp_path):
    from rag import build_index

    tenant_dir = tmp_path / "sis_chunks"
    tenant_dir.mkdir()
    (tenant_dir / "chunks.json").write_text(json.dumps([
        {
            "content": "Enrollment requires proof of residency.",
            "chunk_id": "chunk-1",
            "tenant_id": "sis",
            "doc_id": "doc-1",
            "title": "Enrollment Policy",
            "topic": "enrollment",
            "chunk_index": 0,
            "total_chunks": 1,
            "source_file": "policy.txt",
            "char_count": 39,
            "word_count": 5,
        }
    ]))

    client = _FakeClient(fail_on_staging_add=True)
    monkeypatch.setattr(
        build_index,
        "TENANTS",
        {"sis": SimpleNamespace(chunks_dir=tenant_dir, domain="education")},
    )
    monkeypatch.setattr(build_index, "get_chroma_client", lambda: client)
    monkeypatch.setattr(build_index.time, "time", lambda: 1234.567)

    def fake_embed_texts(texts, show_progress=False):
        return np.ones((len(texts), 3))

    monkeypatch.setattr("rag.embeddings.embed_texts", fake_embed_texts)

    with pytest.raises(ValueError):
        build_index.build_tenant_index("sis")

    assert all("staging" not in name for name in client.collections)
    assert "tenant_sis_docs" not in client.collections
