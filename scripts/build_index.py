"""
Build FAISS Index & Latency Benchmark Script for Phase 4.
1. Loads catalog item embeddings from artifacts/v1/item_embeddings.npy.
2. Builds exact inner product index (IndexFlatIP).
3. Verifies numerical equivalence against brute force NumPy dot product.
4. Benchmarks single-query and batch-query latency (p50, p95, p99).
5. Exports artifacts/v1/faiss.index and reports/retrieval_latency.json.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# Set up project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.reco.indexing.faiss_index import FaissIndex
from src.reco.utils.logging import get_logger
from src.reco.utils.seeds import set_seed

logger = get_logger("build_index")


def build_and_benchmark(version: str = "v1") -> dict:
    """Build FAISS index and run rigorous latency benchmarking."""
    set_seed(42)

    artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts", version)
    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    emb_path = os.path.join(artifacts_dir, "item_embeddings.npy")
    ids_path = os.path.join(artifacts_dir, "item_ids.npy")
    index_path = os.path.join(artifacts_dir, "faiss.index")

    logger.info("Loading catalog item embeddings from %s ...", emb_path)
    item_embeddings = np.load(emb_path)
    item_ids = np.load(ids_path)

    n_items, dim = item_embeddings.shape
    logger.info("Catalog: %d items, dimension: %d", n_items, dim)

    # 1. Build Index
    index = FaissIndex(embedding_dim=dim, index_type="FlatIP")
    index.add(item_embeddings, item_ids)
    index.save(index_path, ids_path)

    # 2. Verify Brute-Force Mathematical Equivalence on 100 Sample Queries
    logger.info("Verifying FAISS vs brute-force dot product on 100 sample queries...")
    rng = np.random.default_rng(42)
    sample_queries = rng.standard_normal((100, dim), dtype=np.float32)
    sample_queries /= np.linalg.norm(sample_queries, axis=1, keepdims=True)

    for i in range(100):
        q = sample_queries[i]
        faiss_res = index.search_single(q, k=20)
        faiss_ids = [r[0] for r in faiss_res]

        brute_scores = np.dot(item_embeddings, q)
        brute_order = np.argsort(-brute_scores)[:20]
        expected_ids = item_ids[brute_order].tolist()

        assert faiss_ids == expected_ids, f"Mismatch at query {i}"

    logger.info("Equivalence check passed: 100/100 queries match brute-force NumPy exactly.")

    # 3. Benchmark Single-Query Latency (1,000 queries)
    logger.info("Benchmarking single-query latency across 1,000 queries...")
    benchmark_queries = rng.standard_normal((1000, dim), dtype=np.float32)
    benchmark_queries /= np.linalg.norm(benchmark_queries, axis=1, keepdims=True)

    latencies_ms = []
    # Warmup
    for i in range(50):
        _ = index.search_single(benchmark_queries[i], k=200)

    for i in range(1000):
        t0 = time.perf_counter()
        _ = index.search_single(benchmark_queries[i], k=200)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    p99 = float(np.percentile(latencies_ms, 99))
    mean_lat = float(np.mean(latencies_ms))

    logger.info(
        "Single-query latency (k=200): p50=%.3f ms, p95=%.3f ms, p99=%.3f ms, mean=%.3f ms",
        p50,
        p95,
        p99,
        mean_lat,
    )

    # 4. Benchmark Batch-Query Latency
    logger.info("Benchmarking batch search throughput...")
    batch_sizes = [1, 8, 32, 64, 128, 256]
    batch_benchmarks = {}

    for bsz in batch_sizes:
        batch_q = rng.standard_normal((bsz, dim), dtype=np.float32)
        batch_q /= np.linalg.norm(batch_q, axis=1, keepdims=True)

        # Warmup
        _ = index.search_batch(batch_q, k=200)

        iters = 50
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = index.search_batch(batch_q, k=200)
        t1 = time.perf_counter()

        total_time = t1 - t0
        time_per_batch_ms = (total_time / iters) * 1000.0
        qps = (bsz * iters) / total_time

        batch_benchmarks[str(bsz)] = {
            "batch_size": bsz,
            "latency_ms": round(time_per_batch_ms, 3),
            "latency_per_query_ms": round(time_per_batch_ms / bsz, 4),
            "queries_per_second": round(qps, 1),
        }
        logger.info(
            "Batch size %d: %.3f ms/batch (%.4f ms/query, %.1f QPS)",
            bsz,
            time_per_batch_ms,
            time_per_batch_ms / bsz,
            qps,
        )

    # 5. Save Benchmark Report
    report = {
        "version": version,
        "index_type": "IndexFlatIP (Exact Inner Product)",
        "num_items": n_items,
        "embedding_dim": dim,
        "index_file": "artifacts/v1/faiss.index",
        "index_file_size_bytes": os.path.getsize(index_path),
        "single_query_latency_ms": {
            "p50": round(p50, 4),
            "p95": round(p95, 4),
            "p99": round(p99, 4),
            "mean": round(mean_lat, 4),
            "min": round(float(np.min(latencies_ms)), 4),
            "max": round(float(np.max(latencies_ms)), 4),
        },
        "batch_benchmarks": batch_benchmarks,
        "brute_force_equivalence": True,
    }

    report_path = os.path.join(reports_dir, "retrieval_latency.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved retrieval latency benchmark to %s", report_path)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS Index & Benchmark")
    parser.add_argument("--version", type=str, default="v1", help="Model artifact version")
    args = parser.parse_args()
    build_and_benchmark(version=args.version)
