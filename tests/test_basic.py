"""
Basic unit tests for CI pipeline.
These tests run without GPU or model loading.
"""
import json
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfig:
    def test_tenant_config_exists(self):
        from tenant_data_pipeline.config import TENANTS
        assert "sis" in TENANTS
        assert "mfg" in TENANTS

    def test_tenant_topics(self):
        from tenant_data_pipeline.config import TENANTS
        assert len(TENANTS["sis"].topics) >= 5
        assert len(TENANTS["mfg"].topics) >= 5

    def test_no_overlapping_topics(self):
        from tenant_data_pipeline.config import TENANTS
        sis_topics = set(TENANTS["sis"].topics)
        mfg_topics = set(TENANTS["mfg"].topics)
        overlap = sis_topics & mfg_topics
        assert len(overlap) == 0, f"Overlapping topics: {overlap}"


class TestGoldenSets:
    def test_sis_golden_set_exists(self):
        path = Path("evaluation/golden_sets/sis_golden.json")
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert len(data) >= 15

    def test_mfg_golden_set_exists(self):
        path = Path("evaluation/golden_sets/mfg_golden.json")
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert len(data) >= 15

    def test_golden_set_format(self):
        for tid in ["sis", "mfg"]:
            path = Path(f"evaluation/golden_sets/{tid}_golden.json")
            with open(path) as f:
                data = json.load(f)

            for item in data:
                assert "id" in item
                assert "question" in item
                assert "expected_answer" in item
                assert len(item["question"]) > 10
                assert len(item["expected_answer"]) > 10

    def test_cross_domain_tests_exist(self):
        for tid in ["sis", "mfg"]:
            path = Path(f"evaluation/golden_sets/{tid}_golden.json")
            with open(path) as f:
                data = json.load(f)

            cross_domain = [d for d in data if d.get("test_type") == "out_of_domain"]
            assert len(cross_domain) >= 1, f"No cross-domain tests for {tid}"


class TestEvaluation:
    def test_keyword_overlap(self):
        from evaluation.eval_config import compute_keyword_overlap
        assert compute_keyword_overlap(
            "proof of residency and birth certificate",
            ["proof of residency", "birth certificate"]
        ) == 1.0
        assert compute_keyword_overlap("hello world", ["proof of residency"]) == 0.0

    def test_hallucination_checker(self):
        from evaluation.hallucination_checker import compute_grounding_score
        context = ["Students must provide proof of residency."]
        good = "Students need to provide proof of residency."
        bad = "Students must pay $500 to a special fund."

        good_score = compute_grounding_score(good, context)
        bad_score = compute_grounding_score(bad, context)
        assert good_score > bad_score

    def test_red_team_has_both_tenants(self):
        from evaluation.red_team import ADVERSARIAL_TESTS
        assert "sis" in ADVERSARIAL_TESTS
        assert "mfg" in ADVERSARIAL_TESTS
        assert len(ADVERSARIAL_TESTS["sis"]) >= 5
        assert len(ADVERSARIAL_TESTS["mfg"]) >= 5

    def test_compliance_tests_exist(self):
        from evaluation.compliance_test import COMPLIANCE_TESTS
        assert "sis" in COMPLIANCE_TESTS
        assert "mfg" in COMPLIANCE_TESTS


class TestMonitoring:
    def test_alert_rules_defined(self):
        from monitoring.alerting import ALERT_RULES
        assert len(ALERT_RULES) >= 4

        categories = {r.category for r in ALERT_RULES}
        assert "performance" in categories
        assert "quality" in categories

    def test_system_metrics_collect(self):
        from monitoring.metrics_collector import MetricsCollector
        mc = MetricsCollector(audit_db_path="/tmp/nonexistent_test.db")
        metrics = mc.collect_system_metrics()
        assert metrics.cpu_percent >= 0
        assert metrics.memory_percent > 0
        assert metrics.memory_used_gb > 0

    def test_tenant_metrics_empty_db(self):
        from monitoring.metrics_collector import MetricsCollector
        mc = MetricsCollector(audit_db_path="/tmp/nonexistent_test.db")
        metrics = mc.collect_tenant_metrics("sis", hours=24)
        assert metrics.total_requests == 0
        assert metrics.tenant_id == "sis"

    def test_registry_operations(self, tmp_path):
        from mlops.registry import ModelRegistry
        reg = ModelRegistry(str(tmp_path / "registry.json"))

        model_id = reg.register_model(
            tenant_id="sis",
            model_type="sft",
            adapter_path="models/adapters/sis/sft",
            metrics={"train_loss": 0.85, "eval_loss": 1.1},
            version="v_test_001",
        )
        assert model_id

        models = reg.list_models("sis")
        assert len(models) == 1
        assert models[0]["version"] == "v_test_001"

        reg.promote_to_production("sis", "sft", "v_test_001")
        active = reg.get_active_model("sis", "sft")
        assert active is not None
        assert active["status"] == "production"


class TestDataPipeline:
    def test_chunker_basics(self):
        from tenant_data_pipeline.chunker import split_text_into_chunks
        sentence = "Students must complete the enrollment process before the semester begins."
        text = " ".join([sentence] * 10)  # ~730 chars with sentence boundaries
        chunks = split_text_into_chunks(text, chunk_size=200, overlap=50)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) >= 50

    def test_synthetic_data_structure(self):
        from tenant_data_pipeline.synthetic_data_generator import SIS_DOCUMENTS, MFG_DOCUMENTS
        assert len(SIS_DOCUMENTS) >= 10
        assert len(MFG_DOCUMENTS) >= 10

        for doc in SIS_DOCUMENTS[:3]:
            assert "title" in doc
            assert "topic" in doc
            assert "content" in doc
            assert len(doc["content"]) > 100

    def test_sft_data_format(self):
        sis_path = Path("data/sis/sft/train_chat.json")
        if sis_path.exists():
            with open(sis_path) as f:
                data = json.load(f)
            assert len(data) > 0
            for item in data[:5]:
                assert "messages" in item
                messages = item["messages"]
                roles = [m["role"] for m in messages]
                assert "user" in roles
                assert "assistant" in roles
