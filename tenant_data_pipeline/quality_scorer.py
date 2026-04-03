"""
Data quality scoring framework.
Produces a quality report per tenant covering:
- Completeness, duplication, topic coverage, PII, bias indicators
"""
import json
import hashlib
from pathlib import Path
from typing import Dict, List
from collections import Counter
from datetime import datetime

from loguru import logger

from tenant_data_pipeline.config import TENANTS


def compute_quality_report(tenant_id: str) -> Dict:
    """Generate a comprehensive data quality report for a tenant."""
    config = TENANTS[tenant_id]

    report: Dict = {
        "tenant_id": tenant_id,
        "domain": config.domain,
        "generated_at": datetime.utcnow().isoformat(),
        "scores": {},
        "details": {},
        "flags": [],
        "overall_status": "PENDING",
    }

    # --- Document-level stats ---
    ingested_path = config.processed_dir / "ingested_documents.json"
    if ingested_path.exists():
        with open(ingested_path) as f:
            docs = json.load(f)
        report["details"]["document_count"] = len(docs)
        report["details"]["total_words"] = sum(d.get("word_count", 0) for d in docs)
        report["details"]["avg_doc_length"] = (
            sum(d.get("word_count", 0) for d in docs) / max(len(docs), 1)
        )
    else:
        docs = []
        report["details"]["document_count"] = 0

    # --- Chunk-level stats ---
    chunks_path = config.chunks_dir / "chunks.json"
    if chunks_path.exists():
        with open(chunks_path) as f:
            chunks = json.load(f)
        report["details"]["chunk_count"] = len(chunks)
        report["details"]["avg_chunk_length"] = (
            sum(c.get("char_count", 0) for c in chunks) / max(len(chunks), 1)
        )
    else:
        chunks = []
        report["details"]["chunk_count"] = 0

    # --- Duplication detection ---
    if chunks:
        content_hashes = [hashlib.md5(c["content"].encode()).hexdigest() for c in chunks]
        unique_hashes = set(content_hashes)
        dup_count = len(content_hashes) - len(unique_hashes)
        dup_rate = dup_count / max(len(content_hashes), 1)
        report["details"]["duplicate_chunks"] = dup_count
        report["details"]["duplication_rate"] = round(dup_rate, 4)
        report["scores"]["duplication"] = round(1.0 - dup_rate, 2)
    else:
        report["scores"]["duplication"] = 0.0

    # --- Topic coverage ---
    expected_topics = set(config.topics)
    if chunks:
        covered_topics = {c["topic"] for c in chunks}
        coverage = len(covered_topics & expected_topics) / max(len(expected_topics), 1)
        missing_topics = expected_topics - covered_topics
        report["details"]["expected_topics"] = sorted(expected_topics)
        report["details"]["covered_topics"] = sorted(covered_topics)
        report["details"]["missing_topics"] = sorted(missing_topics)
        report["scores"]["topic_coverage"] = round(coverage, 2)
        if missing_topics:
            report["flags"].append(f"Missing topics: {sorted(missing_topics)}")
    else:
        report["scores"]["topic_coverage"] = 0.0

    # --- Topic balance ---
    if chunks:
        topic_counts = Counter(c["topic"] for c in chunks)
        report["details"]["chunks_per_topic"] = dict(topic_counts)
        values = list(topic_counts.values())
        balance = min(values) / max(values) if len(values) > 1 else 1.0
        report["scores"]["topic_balance"] = round(balance, 2)
    else:
        report["scores"]["topic_balance"] = 0.0

    # --- PII status ---
    pii_path = config.processed_dir / "pii_report.json"
    if pii_path.exists():
        with open(pii_path) as f:
            pii_report = json.load(f)
        pii_count = pii_report.get("total_pii_found", 0)
        report["details"]["pii_instances_detected"] = pii_count
        report["details"]["pii_by_type"] = pii_report.get("pii_by_type", {})
        report["scores"]["pii_clean"] = 1.0 if pii_count == 0 else max(0.0, 1.0 - pii_count * 0.05)
        if pii_count > 0:
            report["flags"].append(f"PII detected: {pii_count} instances (redacted)")
    else:
        report["scores"]["pii_clean"] = 0.5
        report["flags"].append("PII scan not run")

    # --- Completeness score ---
    doc_count = report["details"]["document_count"]
    completeness = min(1.0, doc_count / 10.0)
    report["scores"]["completeness"] = round(completeness, 2)
    if doc_count < 5:
        report["flags"].append(f"Low document count: {doc_count}")

    # --- Chunk quality (min length check) ---
    if chunks:
        short_chunks = [c for c in chunks if c.get("char_count", 0) < 50]
        report["details"]["short_chunks"] = len(short_chunks)
        chunk_quality = 1.0 - (len(short_chunks) / max(len(chunks), 1))
        report["scores"]["chunk_quality"] = round(chunk_quality, 2)
    else:
        report["scores"]["chunk_quality"] = 0.0

    # --- Bias indicators (keyword-based) ---
    bias_keywords = [
        "always", "never", "all students", "every student", "boys", "girls",
        "disability", "disabled", "poor", "rich", "stupid", "smart", "lazy", "criminal",
    ]
    if chunks:
        bias_hits = []
        for c in chunks:
            content_lower = c["content"].lower()
            for kw in bias_keywords:
                if kw in content_lower:
                    bias_hits.append({"chunk_id": c["chunk_id"], "keyword": kw})
        report["details"]["bias_keyword_hits"] = len(bias_hits)
        report["details"]["bias_keywords_found"] = list({h["keyword"] for h in bias_hits})
        if len(bias_hits) > 10:
            report["flags"].append(f"High bias-sensitive keyword count: {len(bias_hits)} hits")
    else:
        report["details"]["bias_keyword_hits"] = 0

    # --- Overall score ---
    scores = report["scores"]
    if scores:
        overall = sum(scores.values()) / len(scores)
        report["scores"]["overall"] = round(overall, 2)
        report["overall_status"] = (
            "APPROVED" if overall >= 0.7 else "REVIEW_REQUIRED" if overall >= 0.5 else "REJECTED"
        )
    else:
        report["overall_status"] = "NO_DATA"

    report_dir = Path("evaluation/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"data_quality_{tenant_id}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info(
        f"[{tenant_id}] Quality report: overall={report['scores'].get('overall', 'N/A')}, "
        f"status={report['overall_status']}"
    )
    return report


def generate_all_reports() -> Dict:
    results = {}
    for tenant_id in TENANTS:
        results[tenant_id] = compute_quality_report(tenant_id)
    return results


if __name__ == "__main__":
    reports = generate_all_reports()
    for tid, report in reports.items():
        print(f"\n{'='*60}")
        print(f"Tenant: {tid} | Status: {report['overall_status']}")
        for k, v in report["scores"].items():
            print(f"  {k}: {v}")
        if report["flags"]:
            print(f"  Flags: {report['flags']}")
