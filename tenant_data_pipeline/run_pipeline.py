"""
End-to-end data pipeline orchestrator.
Runs all pipeline stages in sequence for all tenants.
"""
import json
import time
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


def run_full_pipeline():
    """Execute the complete data pipeline."""
    logger.info("=" * 70)
    logger.info("MULTI-TENANT DATA PIPELINE — STARTING")
    logger.info("=" * 70)

    pipeline_start = time.time()
    results = {
        "pipeline_run_id": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
        "started_at": datetime.utcnow().isoformat(),
        "stages": {},
    }

    # Stage 1: Generate synthetic data
    logger.info("\nStage 1: Generating synthetic tenant documents...")
    t0 = time.time()
    gen_results = save_synthetic_documents()
    results["stages"]["1_synthetic_generation"] = {
        "status": "success",
        "documents": gen_results,
        "duration_sec": round(time.time() - t0, 2),
    }

    # Stage 2: Ingest documents
    logger.info("\nStage 2: Ingesting documents...")
    t0 = time.time()
    ingest_results = ingest_all_tenants()
    results["stages"]["2_ingestion"] = {
        "status": "success",
        "documents_per_tenant": ingest_results,
        "duration_sec": round(time.time() - t0, 2),
    }

    # Stage 3: PII detection and redaction
    logger.info("\nStage 3: PII detection and redaction...")
    t0 = time.time()
    pii_results = process_all_tenants_pii()
    results["stages"]["3_pii_redaction"] = {
        "status": "success",
        "pii_per_tenant": {
            tid: r.get("total_pii_found", 0) if isinstance(r, dict) else 0
            for tid, r in pii_results.items()
        },
        "duration_sec": round(time.time() - t0, 2),
    }

    # Stage 4: Chunking
    logger.info("\nStage 4: Chunking documents...")
    t0 = time.time()
    chunk_results = chunk_all_tenants()
    results["stages"]["4_chunking"] = {
        "status": "success",
        "chunks_per_tenant": chunk_results,
        "duration_sec": round(time.time() - t0, 2),
    }

    # Stage 5: Data quality scoring
    logger.info("\nStage 5: Data quality assessment...")
    t0 = time.time()
    quality_results = generate_all_reports()
    results["stages"]["5_quality_scoring"] = {
        "status": "success",
        "quality_status": {
            tid: r.get("overall_status", "UNKNOWN")
            for tid, r in quality_results.items()
        },
        "overall_scores": {
            tid: r.get("scores", {}).get("overall", 0)
            for tid, r in quality_results.items()
        },
        "duration_sec": round(time.time() - t0, 2),
    }

    # Stage 6: SFT dataset creation
    logger.info("\nStage 6: Building SFT datasets...")
    t0 = time.time()
    sft_results = build_all_sft_datasets()
    results["stages"]["6_sft_dataset"] = {
        "status": "success",
        "examples_per_tenant": sft_results,
        "duration_sec": round(time.time() - t0, 2),
    }

    # Stage 7: DPO dataset creation
    logger.info("\nStage 7: Building DPO preference datasets...")
    t0 = time.time()
    dpo_results = build_all_dpo_datasets()
    results["stages"]["7_dpo_dataset"] = {
        "status": "success",
        "pairs_per_tenant": dpo_results,
        "duration_sec": round(time.time() - t0, 2),
    }

    # Pipeline summary
    total_duration = round(time.time() - pipeline_start, 2)
    results["completed_at"] = datetime.utcnow().isoformat()
    results["total_duration_sec"] = total_duration

    # Save pipeline results
    report_dir = Path("evaluation/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"pipeline_run_{results['pipeline_run_id']}.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total duration: {total_duration}s")
    for stage_name, stage_data in results["stages"].items():
        logger.info(f"  {stage_name}: {stage_data['status']} ({stage_data['duration_sec']}s)")
    logger.info(f"Report saved: {report_path}")

    return results


if __name__ == "__main__":
    run_full_pipeline()
