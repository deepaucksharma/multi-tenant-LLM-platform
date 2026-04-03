"""
Main evaluation runner.
Runs the golden test set against a model and produces a comprehensive report.

Usage:
    python evaluation/eval_runner.py
    python evaluation/eval_runner.py --tenant sis
"""
import json
import time
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Callable
from datetime import datetime

from loguru import logger

from evaluation.eval_config import (
    EvalResult, EvalReport, load_golden_set,
    compute_keyword_overlap, compute_semantic_similarity,
)
from tenant_data_pipeline.config import TENANTS


def generate_answer_stub(question: str, tenant_id: str) -> str:
    """
    Stub generation function used when no model is loaded.
    Replace with actual model inference in production.
    """
    return f"[STUB] This is a placeholder answer for tenant {tenant_id}."


def run_evaluation(
    tenant_id: str,
    generate_fn: Optional[Callable] = None,
    model_version: str = "baseline",
) -> EvalReport:
    """
    Run full evaluation on a tenant's golden test set.

    Args:
        tenant_id: Tenant to evaluate
        generate_fn: Function(question, tenant_id) -> answer string
        model_version: Model version identifier

    Returns:
        EvalReport
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATION — {tenant_id.upper()} — {model_version}")
    logger.info(f"{'='*60}")

    golden_set = load_golden_set(tenant_id)
    generate = generate_fn or generate_answer_stub

    results = []
    scores_by_category = {}
    total_keyword_score = 0.0
    total_semantic_score = 0.0
    passed_count = 0

    for item in golden_set:
        test_id = item["id"]
        question = item["question"]
        expected = item["expected_answer"]
        required_elements = item.get("required_elements", [])
        category = item.get("category", "general")
        test_type = item.get("test_type", "in_domain")

        # Generate answer
        t0 = time.time()
        try:
            model_answer = generate(question, tenant_id)
        except Exception as e:
            model_answer = f"[ERROR: {e}]"
        latency_ms = round((time.time() - t0) * 1000, 2)

        # Score: keyword overlap
        keyword_score = compute_keyword_overlap(model_answer, required_elements)

        # Score: semantic similarity
        semantic_score = compute_semantic_similarity(model_answer, expected)

        # Score: response length adequacy
        length_ratio = len(model_answer) / max(len(expected), 1)
        length_score = min(1.0, length_ratio) if length_ratio > 0.2 else 0.0

        # Score: out-of-domain handling
        if test_type == "out_of_domain":
            refusal_indicators = [
                "don't have information", "don't have info",
                "not have information", "cannot answer",
                "outside my", "not within my",
                "my expertise covers", "my knowledge covers",
                "consult the appropriate", "direct.*question",
            ]
            refusal_score = 1.0 if any(
                re.search(pat, model_answer, re.IGNORECASE) for pat in refusal_indicators
            ) else 0.0
        else:
            refusal_score = None

        # Combined score
        if test_type == "out_of_domain":
            combined_score = refusal_score
        else:
            combined_score = (keyword_score * 0.4 + semantic_score * 0.4 + length_score * 0.2)

        passed = combined_score >= 0.5

        flags = []
        if keyword_score < 0.5 and test_type != "out_of_domain":
            flags.append("missing_key_elements")
        if semantic_score < 0.3 and test_type != "out_of_domain":
            flags.append("low_semantic_similarity")
        if length_ratio < 0.2:
            flags.append("too_short")
        if length_ratio > 5.0:
            flags.append("too_long")
        if test_type == "out_of_domain" and refusal_score == 0.0:
            flags.append("failed_out_of_domain_refusal")

        result = EvalResult(
            test_id=test_id,
            tenant_id=tenant_id,
            category=category,
            question=question,
            expected_answer=expected,
            model_answer=model_answer,
            scores={
                "keyword_overlap": round(keyword_score, 3),
                "semantic_similarity": round(semantic_score, 3),
                "length_adequacy": round(length_score, 3),
                "combined": round(combined_score, 3),
            },
            passed=passed,
            flags=flags,
            metadata={
                "latency_ms": latency_ms,
                "test_type": test_type,
                "difficulty": item.get("difficulty", "unknown"),
                "required_elements": required_elements,
            },
        )
        results.append(result)

        if passed:
            passed_count += 1
        total_keyword_score += keyword_score
        total_semantic_score += semantic_score

        # Track per-category
        if category not in scores_by_category:
            scores_by_category[category] = {"total": 0, "score_sum": 0.0, "passed": 0}
        scores_by_category[category]["total"] += 1
        scores_by_category[category]["score_sum"] += combined_score
        scores_by_category[category]["passed"] += int(passed)

        status = "PASS" if passed else "FAIL"
        logger.info(
            f"  [{status}] {test_id} [{category}] "
            f"combined={combined_score:.2f} keyword={keyword_score:.2f} "
            f"semantic={semantic_score:.2f}"
        )

    # Build report
    n = len(results)
    avg_keyword = total_keyword_score / max(n, 1)
    avg_semantic = total_semantic_score / max(n, 1)
    pass_rate = passed_count / max(n, 1)

    category_summary = {}
    for cat, data in scores_by_category.items():
        category_summary[cat] = {
            "total": data["total"],
            "passed": data["passed"],
            "avg_score": round(data["score_sum"] / max(data["total"], 1), 3),
            "pass_rate": round(data["passed"] / max(data["total"], 1), 3),
        }

    report = EvalReport(
        report_id=datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        tenant_id=tenant_id,
        model_version=model_version,
        eval_type="golden_set",
        timestamp=datetime.utcnow().isoformat(),
        total_tests=n,
        passed=passed_count,
        failed=n - passed_count,
        pass_rate=round(pass_rate, 3),
        scores={
            "avg_keyword_overlap": round(avg_keyword, 3),
            "avg_semantic_similarity": round(avg_semantic, 3),
            "pass_rate": round(pass_rate, 3),
            "category_breakdown": category_summary,
        },
        results=[
            {
                "test_id": r.test_id,
                "category": r.category,
                "question": r.question[:100],
                "passed": r.passed,
                "scores": r.scores,
                "flags": r.flags,
                "latency_ms": r.metadata.get("latency_ms"),
            }
            for r in results
        ],
        summary=(
            f"{tenant_id.upper()} evaluation: {passed_count}/{n} passed "
            f"({pass_rate:.1%}), avg keyword={avg_keyword:.2f}, "
            f"avg semantic={avg_semantic:.2f}"
        ),
    )

    report.save()

    logger.info(f"\n{report.summary}")
    logger.info(f"Category breakdown:")
    for cat, info in category_summary.items():
        logger.info(f"  {cat}: {info['passed']}/{info['total']} ({info['pass_rate']:.0%})")

    return report


def run_all_evaluations(
    generate_fn: Optional[Callable] = None,
    model_version: str = "baseline",
) -> Dict[str, EvalReport]:
    """Run evaluation for all tenants."""
    reports = {}
    for tenant_id in TENANTS:
        try:
            reports[tenant_id] = run_evaluation(tenant_id, generate_fn, model_version)
        except FileNotFoundError as e:
            logger.warning(f"Skipping {tenant_id}: {e}")
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument("--tenant", type=str, default=None, choices=["sis", "mfg"])
    parser.add_argument("--model-version", type=str, default="stub_baseline")
    args = parser.parse_args()

    if args.tenant:
        run_evaluation(args.tenant, model_version=args.model_version)
    else:
        run_all_evaluations(model_version=args.model_version)
