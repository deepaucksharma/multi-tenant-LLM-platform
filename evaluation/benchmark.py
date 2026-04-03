"""
Performance benchmarking.
Measures latency, throughput, and resource utilization.

Usage:
    python evaluation/benchmark.py
"""
import time
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime

import psutil
from loguru import logger


@dataclass
class BenchmarkResult:
    name: str
    total_requests: int
    successful: int
    failed: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    latency_min_ms: float
    latency_max_ms: float
    throughput_rps: float
    total_time_sec: float
    memory_mb: float
    cpu_percent: float


def measure_latency(
    func: Callable,
    args: tuple = (),
    kwargs: dict = None,
    num_requests: int = 20,
    warmup: int = 2,
) -> BenchmarkResult:
    """
    Measure function latency over multiple calls.

    Args:
        func: Function to benchmark
        args: Positional arguments
        kwargs: Keyword arguments
        num_requests: Number of requests to measure
        warmup: Number of warmup calls (not measured)
    """
    kwargs = kwargs or {}

    # Warmup
    for _ in range(warmup):
        try:
            func(*args, **kwargs)
        except Exception:
            pass

    # Measure
    latencies = []
    successes = 0
    failures = 0
    memory_before = psutil.Process().memory_info().rss / 1024 / 1024

    t_start = time.time()
    for i in range(num_requests):
        t0 = time.time()
        try:
            func(*args, **kwargs)
            successes += 1
        except Exception as e:
            failures += 1
            logger.debug(f"Request {i} failed: {e}")
        latency = (time.time() - t0) * 1000
        latencies.append(latency)

    total_time = time.time() - t_start
    memory_after = psutil.Process().memory_info().rss / 1024 / 1024

    if not latencies:
        latencies = [0.0]

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)

    return BenchmarkResult(
        name=func.__name__ if hasattr(func, '__name__') else str(func),
        total_requests=num_requests,
        successful=successes,
        failed=failures,
        latency_p50_ms=round(sorted_lat[n // 2], 2),
        latency_p95_ms=round(sorted_lat[int(n * 0.95)], 2),
        latency_p99_ms=round(sorted_lat[int(n * 0.99)], 2),
        latency_mean_ms=round(statistics.mean(latencies), 2),
        latency_min_ms=round(min(latencies), 2),
        latency_max_ms=round(max(latencies), 2),
        throughput_rps=round(successes / max(total_time, 0.001), 2),
        total_time_sec=round(total_time, 2),
        memory_mb=round(memory_after, 1),
        cpu_percent=psutil.cpu_percent(interval=0.1),
    )


def run_benchmarks(
    generate_fn: Optional[Callable] = None,
    retrieve_fn: Optional[Callable] = None,
) -> Dict:
    """Run comprehensive performance benchmarks."""
    logger.info(f"\n{'='*60}")
    logger.info("PERFORMANCE BENCHMARKS")
    logger.info(f"{'='*60}")

    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "system_info": {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": round(psutil.virtual_memory().total / 1024**3, 1),
            "memory_available_gb": round(psutil.virtual_memory().available / 1024**3, 1),
        },
        "benchmarks": {},
    }

    # Benchmark 1: Embedding
    logger.info("\nBenchmark: Embedding")
    try:
        from rag.embeddings import embed_query
        embed_result = measure_latency(embed_query, args=("What is the enrollment process?",), num_requests=50)
        results["benchmarks"]["embedding"] = asdict(embed_result)
        logger.info(f"  P50: {embed_result.latency_p50_ms}ms | P95: {embed_result.latency_p95_ms}ms | RPS: {embed_result.throughput_rps}")
    except Exception as e:
        logger.warning(f"  Embedding benchmark failed: {e}")

    # Benchmark 2: Retrieval
    if retrieve_fn:
        logger.info("\nBenchmark: Retrieval")
        retrieve_result = measure_latency(retrieve_fn, args=("enrollment process", "sis"), num_requests=30)
        results["benchmarks"]["retrieval"] = asdict(retrieve_result)
        logger.info(f"  P50: {retrieve_result.latency_p50_ms}ms | P95: {retrieve_result.latency_p95_ms}ms")
    else:
        try:
            from rag.retriever import retrieve
            retrieve_result = measure_latency(retrieve, args=("enrollment process", "sis"), num_requests=30)
            results["benchmarks"]["retrieval"] = asdict(retrieve_result)
            logger.info(f"  P50: {retrieve_result.latency_p50_ms}ms | P95: {retrieve_result.latency_p95_ms}ms")
        except Exception as e:
            logger.warning(f"  Retrieval benchmark failed: {e}")

    # Benchmark 3: Generation (if available)
    if generate_fn:
        logger.info("\nBenchmark: Generation")
        gen_result = measure_latency(
            generate_fn,
            args=("What are the enrollment requirements?", "sis"),
            num_requests=10,
        )
        results["benchmarks"]["generation"] = asdict(gen_result)
        logger.info(f"  P50: {gen_result.latency_p50_ms}ms | P95: {gen_result.latency_p95_ms}ms")

    # Benchmark 4: BM25 search
    logger.info("\nBenchmark: BM25 Search")
    try:
        from rag.bm25_index import search_bm25
        bm25_result = measure_latency(search_bm25, args=("sis", "enrollment process"), num_requests=100)
        results["benchmarks"]["bm25_search"] = asdict(bm25_result)
        logger.info(f"  P50: {bm25_result.latency_p50_ms}ms | P95: {bm25_result.latency_p95_ms}ms")
    except Exception as e:
        logger.warning(f"  BM25 benchmark failed: {e}")

    # Save results
    report_dir = Path("evaluation/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"benchmark_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"\nBenchmark report saved: {report_path}")

    return results


if __name__ == "__main__":
    run_benchmarks()
