"""
Regression testing suite.
Compares model versions to ensure new training doesn't degrade capabilities.

Usage:
    python evaluation/regression_test.py
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Callable
from datetime import datetime

from loguru import logger

from evaluation.eval_config import EvalReport, load_golden_set, compute_semantic_similarity


def run_regression_test(
    tenant_id: str,
    baseline_results_path: str,
    generate_fn: Optional[Callable] = None,
    model_version: str = "candidate",
    regression_threshold: float = 0.05,
) -> EvalReport:
    """
    Compare a new model against baseline results.

    Args:
        tenant_id: Tenant to test
        baseline_results_path: Path to baseline evaluation report JSON
        generate_fn: Generation function for new model
        model_version: New model version identifier
        regression_threshold: Maximum allowed score drop
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"REGRESSION TEST — {tenant_id.upper()} — {model_version} vs baseline")
    logger.info(f"{'='*60}")

    # Load baseline
    baseline_path = Path(baseline_results_path)
    if not baseline_path.exists():
        logger.warning(f"No baseline found at {baseline_path}. Creating initial baseline.")
        return _create_initial_baseline(tenant_id, generate_fn, model_version)

    with open(baseline_path) as f:
        baseline = json.load(f)

    baseline_scores = baseline.get("scores", {})
    baseline_pass_rate = baseline_scores.get("pass_rate", 0.0)

    # Run current model evaluation
    golden_set = load_golden_set(tenant_id)
    results = []
    regressions = []
    improvements = []

    for item in golden_set:
        if generate_fn:
            response = generate_fn(item["question"], tenant_id)
        else:
            response = f"[STUB] Standard answer for {item['category']}"

        # Find baseline result for this test
        baseline_result = None
        for br in baseline.get("results", []):
            if br.get("test_id") == item["id"]:
                baseline_result = br
                break

        current_score = compute_semantic_similarity(response, item["expected_answer"])
        baseline_score = baseline_result.get("scores", {}).get("combined", 0.5) if baseline_result else 0.5

        score_diff = current_score - baseline_score
        is_regression = score_diff < -regression_threshold
        is_improvement = score_diff > regression_threshold

        if is_regression:
            regressions.append(item["id"])
        if is_improvement:
            improvements.append(item["id"])

        results.append({
            "test_id": item["id"],
            "category": item.get("category", ""),
            "baseline_score": round(baseline_score, 3),
            "current_score": round(current_score, 3),
            "diff": round(score_diff, 3),
            "status": "regression" if is_regression else "improvement" if is_improvement else "stable",
        })

    n = len(results)
    avg_current = sum(r["current_score"] for r in results) / max(n, 1)
    avg_baseline = sum(r["baseline_score"] for r in results) / max(n, 1)

    regression_count = len(regressions)
    passed = regression_count == 0

    report = EvalReport(
        report_id=datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        tenant_id=tenant_id,
        model_version=model_version,
        eval_type="regression",
        timestamp=datetime.utcnow().isoformat(),
        total_tests=n,
        passed=n - regression_count,
        failed=regression_count,
        pass_rate=round((n - regression_count) / max(n, 1), 3),
        scores={
            "avg_current_score": round(avg_current, 3),
            "avg_baseline_score": round(avg_baseline, 3),
            "score_diff": round(avg_current - avg_baseline, 3),
            "regressions": regression_count,
            "improvements": len(improvements),
            "stable": n - regression_count - len(improvements),
            "regression_gate": "PASS" if passed else "FAIL",
        },
        results=results,
        summary=(
            f"Regression test {tenant_id.upper()}: "
            f"{'PASS' if passed else 'FAIL'} | "
            f"{regression_count} regressions, {len(improvements)} improvements | "
            f"avg diff: {avg_current - avg_baseline:+.3f}"
        ),
    )

    report.save()
    logger.info(f"\n{report.summary}")

    if regressions:
        logger.warning(f"Regressions detected in: {regressions}")
    if improvements:
        logger.info(f"Improvements in: {improvements}")

    return report


def _create_initial_baseline(
    tenant_id: str,
    generate_fn: Optional[Callable],
    model_version: str,
) -> EvalReport:
    """Create initial baseline when none exists."""
    from evaluation.eval_runner import run_evaluation
    report = run_evaluation(tenant_id, generate_fn, model_version)
    # Save as baseline
    baseline_path = Path(f"evaluation/reports/baseline_{tenant_id}.json")
    baseline_path.write_text(json.dumps(report.to_dict(), indent=2, default=str))
    logger.info(f"Initial baseline saved: {baseline_path}")
    return report


if __name__ == "__main__":
    for tid in ["sis", "mfg"]:
        try:
            run_regression_test(
                tid,
                baseline_results_path=f"evaluation/reports/baseline_{tid}.json",
            )
        except FileNotFoundError as e:
            logger.warning(f"Skipping {tid}: {e}")
