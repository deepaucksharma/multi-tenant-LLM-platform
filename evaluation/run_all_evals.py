"""
Master evaluation orchestrator.
Runs all evaluation suites and generates a consolidated report.

Usage:
    python evaluation/run_all_evals.py
    python evaluation/run_all_evals.py --tenant sis
"""
import json
import argparse
from pathlib import Path
from typing import Optional, Callable, Dict
from datetime import datetime

from loguru import logger

from evaluation.eval_runner import run_all_evaluations, run_evaluation
from evaluation.hallucination_checker import run_hallucination_tests
from evaluation.red_team import run_red_team_tests
from evaluation.bias_audit import run_bias_audit
from evaluation.compliance_test import run_compliance_tests
from evaluation.benchmark import run_benchmarks
from evaluation.judge_llm import run_judge_evaluation
from evaluation.human_eval_protocol import generate_all_forms


def run_complete_evaluation(
    tenant_id: Optional[str] = None,
    generate_fn: Optional[Callable] = None,
    model_version: str = "evaluation_run",
) -> Dict:
    """
    Run the complete evaluation suite.

    Args:
        tenant_id: Specific tenant or None for all
        generate_fn: Model generation function
        model_version: Version string
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"COMPLETE EVALUATION SUITE")
    logger.info(f"Tenant: {tenant_id or 'ALL'} | Model: {model_version}")
    logger.info(f"{'='*70}")

    results = {
        "run_id": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        "model_version": model_version,
        "tenant": tenant_id or "all",
        "timestamp": datetime.utcnow().isoformat(),
        "suites": {},
    }

    # 1. Golden set evaluation
    logger.info("\nSuite 1: Golden Set Evaluation")
    try:
        if tenant_id:
            golden_report = run_evaluation(tenant_id, generate_fn, model_version)
            results["suites"]["golden_set"] = {tenant_id: golden_report.to_dict()}
        else:
            golden_reports = run_all_evaluations(generate_fn, model_version)
            results["suites"]["golden_set"] = {
                tid: r.to_dict() for tid, r in golden_reports.items()
            }
    except Exception as e:
        logger.error(f"Golden set evaluation failed: {e}")
        results["suites"]["golden_set"] = {"error": str(e)}

    # 2. Hallucination tests
    logger.info("\nSuite 2: Hallucination Detection")
    try:
        hall_reports = run_hallucination_tests(tenant_id)
        results["suites"]["hallucination"] = {
            tid: r.to_dict() for tid, r in hall_reports.items()
        }
    except Exception as e:
        logger.error(f"Hallucination tests failed: {e}")
        results["suites"]["hallucination"] = {"error": str(e)}

    # 3. Red team tests
    logger.info("\nSuite 3: Red Team / Adversarial")
    try:
        rt_reports = run_red_team_tests(tenant_id, generate_fn)
        results["suites"]["red_team"] = {
            tid: r.to_dict() for tid, r in rt_reports.items()
        }
    except Exception as e:
        logger.error(f"Red team tests failed: {e}")
        results["suites"]["red_team"] = {"error": str(e)}

    # 4. Bias audit
    logger.info("\nSuite 4: Bias & Fairness Audit")
    try:
        bias_reports = run_bias_audit(tenant_id, generate_fn)
        results["suites"]["bias_audit"] = {
            tid: r.to_dict() for tid, r in bias_reports.items()
        }
    except Exception as e:
        logger.error(f"Bias audit failed: {e}")
        results["suites"]["bias_audit"] = {"error": str(e)}

    # 5. Compliance tests
    logger.info("\nSuite 5: Compliance & Safety")
    try:
        comp_reports = run_compliance_tests(tenant_id, generate_fn)
        results["suites"]["compliance"] = {
            tid: r.to_dict() for tid, r in comp_reports.items()
        }
    except Exception as e:
        logger.error(f"Compliance tests failed: {e}")
        results["suites"]["compliance"] = {"error": str(e)}

    # 6. Judge evaluation
    logger.info("\nSuite 6: LLM-as-Judge")
    try:
        tenants = [tenant_id] if tenant_id else ["sis", "mfg"]
        judge_results = {}
        for tid in tenants:
            judge_report = run_judge_evaluation(tid, generate_fn, model_version)
            judge_results[tid] = judge_report.to_dict()
        results["suites"]["judge"] = judge_results
    except Exception as e:
        logger.error(f"Judge evaluation failed: {e}")
        results["suites"]["judge"] = {"error": str(e)}

    # 7. Benchmarks
    logger.info("\nSuite 7: Performance Benchmarks")
    try:
        bench_results = run_benchmarks(generate_fn)
        results["suites"]["benchmarks"] = bench_results
    except Exception as e:
        logger.error(f"Benchmarks failed: {e}")
        results["suites"]["benchmarks"] = {"error": str(e)}

    # 8. Generate human eval forms
    logger.info("\nSuite 8: Human Evaluation Forms")
    try:
        generate_all_forms()
        results["suites"]["human_eval"] = {"status": "forms_generated"}
    except Exception as e:
        logger.error(f"Human eval form generation failed: {e}")
        results["suites"]["human_eval"] = {"error": str(e)}

    # Consolidated summary
    logger.info(f"\n{'='*70}")
    logger.info("EVALUATION SUMMARY")
    logger.info(f"{'='*70}")

    for suite_name, suite_data in results["suites"].items():
        if isinstance(suite_data, dict) and "error" in suite_data:
            logger.info(f"  {suite_name}: ERROR — {suite_data['error']}")
        elif isinstance(suite_data, dict):
            for tid, report_data in suite_data.items():
                if isinstance(report_data, dict) and "pass_rate" in report_data:
                    pr = report_data["pass_rate"]
                    status = "PASS" if pr >= 0.7 else "WARN" if pr >= 0.5 else "FAIL"
                    logger.info(f"  {suite_name}/{tid}: [{status}] {pr:.0%}")
                elif isinstance(report_data, dict) and "summary" in report_data:
                    logger.info(f"  {suite_name}/{tid}: {report_data['summary']}")
                else:
                    logger.info(f"  {suite_name}/{tid}: completed")

    # Save consolidated report
    report_dir = Path("evaluation/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"consolidated_{results['run_id']}.json"
    report_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"\nConsolidated report: {report_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete evaluation suite")
    parser.add_argument("--tenant", type=str, default=None, choices=["sis", "mfg"])
    parser.add_argument("--model-version", type=str, default="stub_baseline")
    args = parser.parse_args()

    run_complete_evaluation(
        tenant_id=args.tenant,
        model_version=args.model_version,
    )
