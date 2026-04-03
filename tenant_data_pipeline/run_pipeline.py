"""
End-to-end data pipeline orchestrator.
Runs all pipeline stages in sequence for all tenants.
"""
import json
import time
import traceback
from pathlib import Path
from datetime import datetime

from loguru import logger

from tenant_data_pipeline.synthetic_data_generator import save_synthetic_documents
from tenant_data_pipeline.ingest import ingest_all_tenants
from tenant_data_pipeline.pii_redact import process_all_tenants_pii
from tenant_data_pipeline.chunker import chunk_all_tenants
from tenant_data_pipeline.quality_scorer import generate_all_reports
from tenant_data_pipeline.sft_data_builder import build_all_sft_datasets
from tenant_data_pipeline.dpo_data_builder import build_all_dpo_datasets


def _write_pipeline_report(results: dict) -> Path:
    """Persist current pipeline status so failures leave a usable report."""
    report_dir = Path("evaluation/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"pipeline_run_{results['pipeline_run_id']}.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return report_path


def _run_stage(results: dict, stage_key: str, message: str, fn, summarize):
    """Execute one stage, recording status and persisting partial progress."""
    logger.info(f"\n{message}")
    t0 = time.time()
    try:
        stage_result = fn()
        results["stages"][stage_key] = {
            "status": "success",
            **summarize(stage_result),
            "duration_sec": round(time.time() - t0, 2),
        }
        _write_pipeline_report(results)
        return stage_result
    except Exception as exc:
        results["stages"][stage_key] = {
            "status": "failed",
            "error": str(exc),
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
            "duration_sec": round(time.time() - t0, 2),
        }
        raise


def run_full_pipeline():
    """Execute the complete data pipeline."""
    logger.info("=" * 70)
    logger.info("MULTI-TENANT DATA PIPELINE — STARTING")
    logger.info("=" * 70)

    pipeline_start = time.time()
    results = {
        "pipeline_run_id": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        "started_at": datetime.utcnow().isoformat(),
        "status": "running",
        "stages": {},
    }
    report_path = _write_pipeline_report(results)

    try:
        _run_stage(
            results,
            "1_synthetic_generation",
            "Stage 1: Generating synthetic tenant documents...",
            save_synthetic_documents,
            lambda gen_results: {"documents": gen_results},
        )
        _run_stage(
            results,
            "2_ingestion",
            "Stage 2: Ingesting documents...",
            ingest_all_tenants,
            lambda ingest_results: {"documents_per_tenant": ingest_results},
        )
        _run_stage(
            results,
            "3_pii_redaction",
            "Stage 3: PII detection and redaction...",
            process_all_tenants_pii,
            lambda pii_results: {
                "pii_per_tenant": {
                    tid: r.get("total_pii_found", 0) if isinstance(r, dict) else 0
                    for tid, r in pii_results.items()
                },
                "compliance_status": {
                    tid: r.get("compliance_status", "unknown") if isinstance(r, dict) else "unknown"
                    for tid, r in pii_results.items()
                },
            },
        )
        _run_stage(
            results,
            "4_chunking",
            "Stage 4: Chunking documents...",
            chunk_all_tenants,
            lambda chunk_results: {"chunks_per_tenant": chunk_results},
        )
        _run_stage(
            results,
            "5_quality_scoring",
            "Stage 5: Data quality assessment...",
            generate_all_reports,
            lambda quality_results: {
                "quality_status": {
                    tid: r.get("overall_status", "UNKNOWN")
                    for tid, r in quality_results.items()
                },
                "overall_scores": {
                    tid: r.get("scores", {}).get("overall", 0)
                    for tid, r in quality_results.items()
                },
            },
        )
        _run_stage(
            results,
            "6_sft_dataset",
            "Stage 6: Building SFT datasets...",
            build_all_sft_datasets,
            lambda sft_results: {"examples_per_tenant": sft_results},
        )
        _run_stage(
            results,
            "7_dpo_dataset",
            "Stage 7: Building DPO preference datasets...",
            build_all_dpo_datasets,
            lambda dpo_results: {"pairs_per_tenant": dpo_results},
        )

        results["status"] = "success"
        results["completed_at"] = datetime.utcnow().isoformat()
        results["total_duration_sec"] = round(time.time() - pipeline_start, 2)
        report_path = _write_pipeline_report(results)

        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total duration: {results['total_duration_sec']}s")
        for stage_name, stage_data in results["stages"].items():
            logger.info(f"  {stage_name}: {stage_data['status']} ({stage_data['duration_sec']}s)")
        logger.info(f"Report saved: {report_path}")

        return results
    except Exception as exc:
        results["status"] = "failed"
        results["failed_stage"] = next(
            (name for name, data in results["stages"].items() if data["status"] == "failed"),
            "unknown",
        )
        results["completed_at"] = datetime.utcnow().isoformat()
        results["total_duration_sec"] = round(time.time() - pipeline_start, 2)
        report_path = _write_pipeline_report(results)
        logger.exception(
            f"Pipeline failed at {results['failed_stage']}. Partial report saved: {report_path}"
        )
        raise RuntimeError(
            f"Pipeline failed at {results['failed_stage']}. "
            f"See {report_path} for partial progress and traceback."
        ) from exc


if __name__ == "__main__":
    run_full_pipeline()
