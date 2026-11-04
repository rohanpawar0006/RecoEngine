"""
Serving Latency Benchmark Script.
Measures p50, p95, p99, and mean latency (in milliseconds) across:
1. Cache Hit (cached warm user recommendation)
2. Warm User Cache Miss (Full Pipeline: Two-Tower retrieval + FAISS + DCN-V2 Ranker)
3. Cold User Popularity Fallback (Time-decayed popularity)
4. Cold User Content Fallback (Content-based genre matching)

Outputs results to reports/serving_latency.json.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

import numpy as np
from fastapi.testclient import TestClient

from src.reco.serving.app import app, load_serving_state, state


def run_benchmark(n_iterations: int = 100) -> dict:
    logger.info("Initializing serving state...")
    load_serving_state()
    client = TestClient(app)

    warm_users = list(state.get("user_features", {}).keys())
    if not warm_users:
        raise RuntimeError("No warm users found in feature store.")

    test_user = warm_users[0]
    cache = state.get("cache")

    logger.info("Warmup queries...")
    for _ in range(5):
        client.get("/health")
        client.post("/recommend", json={"user_id": test_user, "k": 10, "variant": "warmup"})

    results = {}

    # -------------------------------------------------------------
    # 1. Warm User Cache Hit Benchmark
    # -------------------------------------------------------------
    logger.info("Benchmarking Warm User Cache Hit (%d iterations)...", n_iterations)
    hit_variant = "bench_hit"
    # Seed cache
    client.post("/recommend", json={"user_id": test_user, "k": 10, "variant": hit_variant})

    latencies_hit = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        resp = client.post("/recommend", json={"user_id": test_user, "k": 10, "variant": hit_variant})
        dt = (time.perf_counter() - t0) * 1000.0
        assert resp.status_code == 200
        assert resp.json()["cache_hit"] is True
        latencies_hit.append(dt)

    results["warm_cache_hit"] = {
        "n_requests": n_iterations,
        "mean_ms": round(float(np.mean(latencies_hit)), 3),
        "p50_ms": round(float(np.percentile(latencies_hit, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies_hit, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies_hit, 99)), 3),
    }

    # -------------------------------------------------------------
    # 2. Warm User Cache Miss (Full Two-Stage Pipeline) Benchmark
    # -------------------------------------------------------------
    logger.info("Benchmarking Warm User Cache Miss (%d iterations)...", n_iterations)
    # Use distinct users or bust cache to measure pure end-to-end inference
    sample_users = [warm_users[i % len(warm_users)] for i in range(n_iterations)]
    latencies_miss = []
    for i, u in enumerate(sample_users):
        # Invalidate cache for this user
        if cache and cache.is_available():
            cache.client.delete(f"rec:bench_miss:{u}:10")

        t0 = time.perf_counter()
        resp = client.post("/recommend", json={"user_id": u, "k": 10, "variant": "bench_miss"})
        dt = (time.perf_counter() - t0) * 1000.0
        assert resp.status_code == 200
        latencies_miss.append(dt)

    results["warm_cache_miss_full_pipeline"] = {
        "n_requests": n_iterations,
        "mean_ms": round(float(np.mean(latencies_miss)), 3),
        "p50_ms": round(float(np.percentile(latencies_miss, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies_miss, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies_miss, 99)), 3),
    }

    # -------------------------------------------------------------
    # 3. Cold User Popularity Fallback Benchmark
    # -------------------------------------------------------------
    logger.info("Benchmarking Cold User Popularity Fallback (%d iterations)...", n_iterations)
    latencies_cold_pop = []
    for i in range(n_iterations):
        cu = 90000000 + i
        if cache and cache.is_available():
            cache.client.delete(f"rec:bench_cold_pop:{cu}:10")

        t0 = time.perf_counter()
        resp = client.post("/recommend", json={"user_id": cu, "k": 10, "variant": "bench_cold_pop"})
        dt = (time.perf_counter() - t0) * 1000.0
        assert resp.status_code == 200
        assert resp.json()["source"] == "fallback_popularity"
        latencies_cold_pop.append(dt)

    results["cold_popularity_fallback"] = {
        "n_requests": n_iterations,
        "mean_ms": round(float(np.mean(latencies_cold_pop)), 3),
        "p50_ms": round(float(np.percentile(latencies_cold_pop, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies_cold_pop, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies_cold_pop, 99)), 3),
    }

    # -------------------------------------------------------------
    # 4. Cold User Content Fallback Benchmark
    # -------------------------------------------------------------
    logger.info("Benchmarking Cold User Content Fallback (%d iterations)...", n_iterations)
    latencies_cold_content = []
    for i in range(n_iterations):
        cu = 80000000 + i
        if cache and cache.is_available():
            cache.client.delete(f"rec:bench_cold_content:{cu}:10")

        t0 = time.perf_counter()
        resp = client.post(
            "/recommend",
            json={"user_id": cu, "k": 10, "variant": "bench_cold_content", "seed_genres": ["Sci-Fi", "Action"]},
        )
        dt = (time.perf_counter() - t0) * 1000.0
        assert resp.status_code == 200
        assert resp.json()["source"] == "fallback_content"
        latencies_cold_content.append(dt)

    results["cold_content_fallback"] = {
        "n_requests": n_iterations,
        "mean_ms": round(float(np.mean(latencies_cold_content)), 3),
        "p50_ms": round(float(np.percentile(latencies_cold_content, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies_cold_content, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies_cold_content, 99)), 3),
    }

    # Environment & Serving details
    results["metadata"] = {
        "served_by": state.get("ranker").served_by if state.get("ranker") else "none",
        "redis_available": state.get("cache").is_available() if state.get("cache") else False,
        "faiss_loaded": state.get("faiss_index") is not None,
        "catalog_size": state.get("faiss_index").ntotal if state.get("faiss_index") else 0,
    }

    # Save to reports
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "serving_latency.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved serving latency benchmark to %s", out_path)
    logger.info("Summary Results:\n%s", json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run_benchmark(n_iterations=100)
