"""
Hallucination detection pipeline.
Verifies that model responses are grounded in retrieved context.
Detects fabricated claims, unsupported numbers, and invented procedures.

Usage:
    python evaluation/hallucination_checker.py
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime

from loguru import logger

from evaluation.eval_config import EvalReport


@dataclass
class HallucinationTestCase:
    test_id: str
    tenant_id: str
    question: str
    context_chunks: List[str]
    model_response: str
    grounding_score: float = 0.0
    hallucinated_claims: List[str] = field(default_factory=list)
    fabricated_numbers: List[str] = field(default_factory=list)
    unsupported_procedures: List[str] = field(default_factory=list)
    passed: bool = True


# Predefined test cases with known context
HALLUCINATION_TESTS = {
    "sis": [
        {
            "test_id": "hall_sis_001",
            "question": "What is the enrollment process?",
            "context": [
                "Students must provide proof of residency, immunization records, and birth certificate.",
                "The registrar verifies documents within 3 business days.",
                "Transfer students must submit official transcripts.",
            ],
            "good_response": "Students must provide proof of residency, immunization records, and a birth certificate. The registrar verifies these within 3 business days.",
            "bad_response": "Students must provide a $200 enrollment fee, three letters of recommendation, and pass an entrance exam. The principal personally reviews each application within 24 hours.",
        },
        {
            "test_id": "hall_sis_002",
            "question": "How are grades calculated?",
            "context": [
                "Grade weights: Assessments/Tests 40%, Classwork/Participation 25%, Homework 15%, Projects/Presentations 20%.",
                "Grading scale: A 90-100, B 80-89, C 70-79, D 60-69, F Below 60.",
            ],
            "good_response": "Grades are calculated using: Tests 40%, Classwork 25%, Homework 15%, Projects 20%. The grading scale runs from A (90-100) to F (below 60).",
            "bad_response": "Grades are based on a 500-point system where attendance counts for 60% of the grade. Students must maintain a 3.5 GPA to remain enrolled.",
        },
        {
            "test_id": "hall_sis_003",
            "question": "What are the attendance requirements?",
            "context": [
                "Teachers must record attendance within the first 15 minutes of each class period.",
                "After 3 unexcused absences: automated letter sent.",
                "After 5 unexcused absences: counselor contacts family.",
            ],
            "good_response": "Teachers record attendance within 15 minutes of class start. After 3 unexcused absences, an automated letter is sent. After 5, the counselor contacts the family.",
            "bad_response": "Students are allowed 20 unexcused absences per semester. Attendance is recorded weekly by the principal. There are no consequences until the student misses 30 days.",
        },
    ],
    "mfg": [
        {
            "test_id": "hall_mfg_001",
            "question": "What is the defect escalation process?",
            "context": [
                "Class 1: Quality Manager notified within 1 hour. Customer notified within 24 hours.",
                "Class 2: Quality Engineer notified within 4 hours.",
                "Class 3: Logged and reviewed weekly.",
            ],
            "good_response": "Class 1 defects require Quality Manager notification within 1 hour and customer notification within 24 hours. Class 2 defects need Quality Engineer notification within 4 hours.",
            "bad_response": "All defects are reported to corporate headquarters within 30 days. The production line continues running during the investigation. No customer notification is required.",
        },
        {
            "test_id": "hall_mfg_002",
            "question": "What is the preventive maintenance schedule?",
            "context": [
                "Daily: visual inspection, lubrication checks.",
                "Weekly: belt tension, fluid levels.",
                "Monthly: full lubrication, calibration verification.",
                "OEE target above 85%.",
            ],
            "good_response": "PM schedule: daily visual inspection and lubrication, weekly belt tension and fluid checks, monthly full lubrication and calibration. OEE target is above 85%.",
            "bad_response": "Maintenance is performed annually. Machines run continuously without inspection. The facility operates a run-to-failure maintenance strategy with an OEE target of 50%.",
        },
        {
            "test_id": "hall_mfg_003",
            "question": "What safety equipment is required?",
            "context": [
                "Zone A (Assembly): Safety glasses, steel-toe boots, hearing protection, gloves.",
                "Zone B (Welding): Full face shield, welding gloves, fire-resistant clothing.",
                "Maximum lift weight without assistance: 50 lbs.",
            ],
            "good_response": "Assembly zone requires safety glasses, steel-toe boots, hearing protection, and gloves. Welding zone needs full face shield, welding gloves, and fire-resistant clothing. Max unassisted lift is 50 lbs.",
            "bad_response": "No PPE is required in the assembly area. Workers in the welding zone only need sunglasses. There is no weight lifting limit as long as you use proper form.",
        },
    ],
}


def check_number_grounding(response: str, context: List[str]) -> List[str]:
    """Find numbers in response that don't appear in any context chunk."""
    response_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', response))
    context_text = " ".join(context)
    context_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', context_text))

    # Filter out very common numbers (1, 2, etc.)
    trivial = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}
    response_numbers -= trivial
    fabricated = response_numbers - context_numbers - trivial
    return list(fabricated)


def check_procedure_grounding(response: str, context: List[str]) -> List[str]:
    """Detect procedure-like claims not found in context."""
    context_text = " ".join(context).lower()

    procedure_patterns = [
        r'(?:must|required to|need to|should)\s+(.{10,60}?)(?:\.|,|$)',
        r'(?:procedure|process|step|protocol)(?:\s+is)?\s*:\s*(.{10,80}?)(?:\.|,|$)',
    ]

    unsupported = []
    for pattern in procedure_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        for match in matches:
            claim = match.strip().lower()
            # Check if key phrases from the claim appear in context
            claim_words = set(re.findall(r'\b\w{4,}\b', claim))
            context_words = set(re.findall(r'\b\w{4,}\b', context_text))
            overlap = len(claim_words & context_words) / max(len(claim_words), 1)
            if overlap < 0.3:
                unsupported.append(match.strip())

    return unsupported


def compute_grounding_score(response: str, context: List[str]) -> float:
    """Compute overall grounding score."""
    if not context or not response:
        return 0.0

    response_lower = response.lower()
    context_text = " ".join(context).lower()

    # Token overlap
    response_tokens = set(re.findall(r'\b\w{3,}\b', response_lower))
    context_tokens = set(re.findall(r'\b\w{3,}\b', context_text))

    if not response_tokens:
        return 0.0

    overlap = len(response_tokens & context_tokens) / len(response_tokens)

    # Penalize for fabricated numbers
    fab_numbers = check_number_grounding(response, context)
    number_penalty = min(0.3, len(fab_numbers) * 0.1)

    # Penalize for unsupported procedures
    unsupported = check_procedure_grounding(response, context)
    procedure_penalty = min(0.3, len(unsupported) * 0.1)

    score = max(0.0, overlap - number_penalty - procedure_penalty)
    return round(score, 3)


def run_hallucination_tests(tenant_id: str = None) -> Dict[str, EvalReport]:
    """Run hallucination detection tests."""
    reports = {}
    tenants = [tenant_id] if tenant_id else list(HALLUCINATION_TESTS.keys())

    for tid in tenants:
        tests = HALLUCINATION_TESTS.get(tid, [])
        if not tests:
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"HALLUCINATION TESTS — {tid.upper()}")
        logger.info(f"{'='*60}")

        results = []
        passed_count = 0

        for test in tests:
            context = test["context"]

            # Test good response (should be grounded)
            good_score = compute_grounding_score(test["good_response"], context)
            good_fab_numbers = check_number_grounding(test["good_response"], context)
            good_unsupported = check_procedure_grounding(test["good_response"], context)

            # Test bad response (should be detected as hallucination)
            bad_score = compute_grounding_score(test["bad_response"], context)
            bad_fab_numbers = check_number_grounding(test["bad_response"], context)
            bad_unsupported = check_procedure_grounding(test["bad_response"], context)

            # Good response should score higher than bad
            detection_correct = good_score > bad_score
            good_is_grounded = good_score >= 0.4
            bad_is_hallucinated = bad_score < 0.4

            passed = detection_correct and good_is_grounded and bad_is_hallucinated
            if passed:
                passed_count += 1

            status = "PASS" if passed else "FAIL"
            logger.info(
                f"  [{status}] {test['test_id']}: "
                f"good_score={good_score:.2f} bad_score={bad_score:.2f} "
                f"detection={'correct' if detection_correct else 'FAILED'}"
            )

            results.append({
                "test_id": test["test_id"],
                "question": test["question"],
                "good_response_score": good_score,
                "bad_response_score": bad_score,
                "good_fabricated_numbers": good_fab_numbers,
                "bad_fabricated_numbers": bad_fab_numbers,
                "good_unsupported_procedures": good_unsupported,
                "bad_unsupported_procedures": bad_unsupported,
                "detection_correct": detection_correct,
                "passed": passed,
            })

        n = len(results)
        pass_rate = passed_count / max(n, 1)

        report = EvalReport(
            report_id=datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
            tenant_id=tid,
            model_version="hallucination_checker",
            eval_type="hallucination",
            timestamp=datetime.utcnow().isoformat(),
            total_tests=n,
            passed=passed_count,
            failed=n - passed_count,
            pass_rate=round(pass_rate, 3),
            scores={
                "detection_accuracy": round(pass_rate, 3),
                "avg_good_score": round(
                    sum(r["good_response_score"] for r in results) / max(n, 1), 3
                ),
                "avg_bad_score": round(
                    sum(r["bad_response_score"] for r in results) / max(n, 1), 3
                ),
            },
            results=results,
            summary=(
                f"Hallucination detection {tid.upper()}: {passed_count}/{n} "
                f"({pass_rate:.0%}) correct detections"
            ),
        )

        report.save()
        reports[tid] = report
        logger.info(f"\n{report.summary}")

    return reports


if __name__ == "__main__":
    run_hallucination_tests()
