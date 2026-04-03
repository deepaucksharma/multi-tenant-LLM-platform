"""
PII detection and redaction pipeline.
Uses regex patterns + optional Presidio for entity detection.
Critical for FERPA compliance (SIS tenant) and general data governance.
"""
import re
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from loguru import logger

from tenant_data_pipeline.config import TENANTS

try:
    from presidio_analyzer import AnalyzerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    logger.info("Presidio not available, using regex-only PII detection")

_PRESIDIO_ANALYZER: Optional["AnalyzerEngine"] = None


@dataclass
class PIIFinding:
    doc_id: str
    tenant_id: str
    pii_type: str
    original: str
    redacted: str
    start: int
    end: int
    confidence: float


PII_PATTERNS = {
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "PHONE": r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "STUDENT_ID": r"\b(?:STU|SID|ID)[#:\s-]?\d{5,10}\b",
    "DATE_OF_BIRTH": r"\b(?:DOB|Date of Birth|born)[:\s]+\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b",
    "CREDIT_CARD": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "IP_ADDRESS": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
}

REDACTION_MAP = {
    "SSN": "[SSN_REDACTED]",
    "PHONE": "[PHONE_REDACTED]",
    "EMAIL": "[EMAIL_REDACTED]",
    "STUDENT_ID": "[STUDENT_ID_REDACTED]",
    "DATE_OF_BIRTH": "[DOB_REDACTED]",
    "CREDIT_CARD": "[CC_REDACTED]",
    "IP_ADDRESS": "[IP_REDACTED]",
    "PERSON": "[NAME_REDACTED]",
    "LOCATION": "[LOCATION_REDACTED]",
}

PRESIDIO_ENTITY_MAP = {
    "PERSON": "PERSON",
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "US_SSN": "SSN",
    "CREDIT_CARD": "CREDIT_CARD",
    "IP_ADDRESS": "IP_ADDRESS",
    "DATE_TIME": "DATE_OF_BIRTH",
    "LOCATION": "LOCATION",
}

PRESIDIO_REQUIRED_TENANTS = {"sis"}


def detect_pii_regex(text: str, doc_id: str, tenant_id: str) -> List[PIIFinding]:
    """Detect PII using regex patterns."""
    findings = []
    for pii_type, pattern in PII_PATTERNS.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            findings.append(PIIFinding(
                doc_id=doc_id,
                tenant_id=tenant_id,
                pii_type=pii_type,
                original=match.group(),
                redacted=REDACTION_MAP.get(pii_type, "[REDACTED]"),
                start=match.start(),
                end=match.end(),
                confidence=0.85,
            ))
    return findings


def detect_pii_presidio(text: str, doc_id: str, tenant_id: str) -> List[PIIFinding]:
    """Detect PII using Presidio analyzer."""
    if not PRESIDIO_AVAILABLE:
        return []

    global _PRESIDIO_ANALYZER
    if _PRESIDIO_ANALYZER is None:
        _PRESIDIO_ANALYZER = AnalyzerEngine()

    results = _PRESIDIO_ANALYZER.analyze(
        text=text,
        entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN",
                  "CREDIT_CARD", "IP_ADDRESS", "DATE_TIME", "LOCATION"],
        language="en",
    )

    findings = []
    for result in results:
        pii_type = PRESIDIO_ENTITY_MAP.get(result.entity_type, result.entity_type)
        findings.append(PIIFinding(
            doc_id=doc_id,
            tenant_id=tenant_id,
            pii_type=pii_type,
            original=text[result.start:result.end],
            redacted=REDACTION_MAP.get(pii_type, "[REDACTED]"),
            start=result.start,
            end=result.end,
            confidence=result.score,
        ))
    return findings


def redact_text(text: str, findings: List[PIIFinding]) -> str:
    """Apply redactions end-to-start to preserve character positions."""
    sorted_findings = sorted(findings, key=lambda f: f.start, reverse=True)
    redacted = text
    for finding in sorted_findings:
        redacted = redacted[:finding.start] + finding.redacted + redacted[finding.end:]
    return redacted


def get_pii_runtime_status(tenant_id: str) -> Dict[str, object]:
    """Describe the active PII detection engines and compliance caveats."""
    detection_engines = ["regex"]
    warnings: List[str] = []
    compliance_status = "ok"

    if PRESIDIO_AVAILABLE:
        detection_engines.append("presidio")
    elif tenant_id in PRESIDIO_REQUIRED_TENANTS:
        compliance_status = "degraded"
        warnings.append(
            "Presidio is unavailable; only regex PII detection is active. "
            "NLP-based entities such as names and locations will be missed."
        )

    return {
        "presidio_available": PRESIDIO_AVAILABLE,
        "detection_engines": detection_engines,
        "compliance_status": compliance_status,
        "warnings": warnings,
    }


def process_tenant_pii(tenant_id: str) -> Dict:
    """Run PII detection and redaction for all documents of a tenant."""
    config = TENANTS[tenant_id]
    processed_path = config.processed_dir / "ingested_documents.json"
    runtime_status = get_pii_runtime_status(tenant_id)

    for warning in runtime_status["warnings"]:
        logger.warning(f"[{tenant_id}] {warning}")

    if not processed_path.exists():
        logger.warning(f"No ingested documents found for {tenant_id}")
        return {
            "tenant_id": tenant_id,
            "documents_processed": 0,
            "total_pii_found": 0,
            **runtime_status,
        }

    with open(processed_path) as f:
        documents = json.load(f)

    all_findings = []
    redacted_documents = []

    for doc in documents:
        doc_id = doc["doc_id"]
        text = doc["content"]

        findings = detect_pii_regex(text, doc_id, tenant_id)

        if PRESIDIO_AVAILABLE:
            presidio_findings = detect_pii_presidio(text, doc_id, tenant_id)
            findings.extend(presidio_findings)

        findings = _deduplicate_findings(findings)
        all_findings.extend(findings)

        redacted_content = redact_text(text, findings)
        doc["content_redacted"] = redacted_content
        doc["pii_findings_count"] = len(findings)
        doc["pii_types_found"] = list({f.pii_type for f in findings})
        redacted_documents.append(doc)

    redacted_path = config.processed_dir / "redacted_documents.json"
    redacted_path.write_text(json.dumps(redacted_documents, indent=2), encoding="utf-8")

    report = {
        "tenant_id": tenant_id,
        "scan_timestamp": datetime.utcnow().isoformat(),
        "documents_processed": len(documents),
        "total_pii_found": len(all_findings),
        "pii_by_type": _count_by_type(all_findings),
        "findings": [asdict(f) for f in all_findings],
        **runtime_status,
    }
    report_path = config.processed_dir / "pii_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info(
        f"[{tenant_id}] PII scan: {len(documents)} docs, "
        f"{len(all_findings)} PII instances found and redacted"
    )
    return report


def _deduplicate_findings(findings: List[PIIFinding]) -> List[PIIFinding]:
    """Remove overlapping findings, preferring higher-confidence, wider spans."""
    if not findings:
        return findings
    ranked = sorted(
        findings,
        key=lambda f: (-f.confidence, -(f.end - f.start), f.start, f.end),
    )
    selected: List[PIIFinding] = []
    for finding in ranked:
        overlaps = any(
            not (finding.end <= existing.start or finding.start >= existing.end)
            for existing in selected
        )
        if not overlaps:
            selected.append(finding)
    return sorted(selected, key=lambda f: (f.start, f.end))


def _count_by_type(findings: List[PIIFinding]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for f in findings:
        counts[f.pii_type] = counts.get(f.pii_type, 0) + 1
    return counts


def process_all_tenants_pii() -> Dict:
    results = {}
    for tenant_id in TENANTS:
        results[tenant_id] = process_tenant_pii(tenant_id)
    return results


if __name__ == "__main__":
    results = process_all_tenants_pii()
    for tid, report in results.items():
        print(f"[{tid}] PII found: {report.get('total_pii_found', 0)}")
