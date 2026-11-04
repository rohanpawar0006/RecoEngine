"""
Evaluate cold-start fallbacks on held-out cold validation users.
Compares:
1. Random baseline
2. Global popularity baseline
3. Time-decayed popularity fallback
4. Content-based genre fallback (using first liked genre as seed)

Also exports production fallback artifacts to artifacts/v1/ and populates Redis cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Set

import numpy as np
import pandas as pd

from src.reco.evaluation.metrics import evaluate_recommendations
from src.reco.fallback.content_based import ContentBasedRecommender
from src.reco.fallback.popularity import TimeDecayedPopularity
from src.reco.serving.cache import RedisCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Cold-Start Fallbacks")
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/processed/dev/train.parquet",
        help="Path to training interactions Parquet",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default="data/processed/val.parquet",
        help="Path to validation interactions Parquet",
    )
    parser.add_argument(
        "--movies-data",
        type=str,
        default="data/raw/ml-25m/movies.csv",
        help="Path to movies CSV",
    )
    parser.add_argument(
        "--item-features",
        type=str,
        default="data/features/dev/item_features.parquet",
        help="Path to item features Parquet",
    )
    parser.add_argument(
        "--vocab-users",
        type=str,
        default="data/processed/vocab_users.json",
        help="Path to training user vocab JSON",
    )
    parser.add_argument(
        "--max-eval-users",
        type=int,
        default=2000,
        help="Maximum cold-start users to evaluate",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="reports/cold_start_metrics.json",
        help="Path to save evaluation JSON report",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts/v1",
        help="Directory to save fallback artifacts",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    logger.info("Loading training and validation data...")
    train_df = pd.read_parquet(args.train_data)
    val_df = pd.read_parquet(args.val_data)
    movies_df = pd.read_csv(args.movies_data)
    item_features_df = (
        pd.read_parquet(args.item_features) if Path(args.item_features).exists() else None
    )

    with open(args.vocab_users, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
        vocab_users = set(vocab_data.get("id_to_index", {}).keys())

    # 2. Identify cold users in validation set (never seen in train vocabulary)
    val_users = set(val_df["userId"].unique())
    cold_user_ids = sorted(list(val_users - {int(u) for u in vocab_users if u != "-1"}))
    logger.info(
        "Identified %d cold users out of %d validation users (%.1f%% cold).",
        len(cold_user_ids),
        len(val_users),
        100.0 * len(cold_user_ids) / len(val_users) if val_users else 0.0,
    )

    # Filter to cold users with positive interactions
    cold_positives = val_df[(val_df["userId"].isin(cold_user_ids)) & (val_df["is_positive"] == 1)]
    valid_cold_users = sorted(cold_positives["userId"].unique())
    logger.info("Cold users with at least 1 positive interaction: %d", len(valid_cold_users))

    if len(valid_cold_users) > args.max_eval_users:
        random.seed(42)
        eval_cold_users = random.sample(valid_cold_users, args.max_eval_users)
    else:
        eval_cold_users = valid_cold_users

    # Ground truth positives for evaluation users
    eval_positives_df = cold_positives[cold_positives["userId"].isin(eval_cold_users)]
    user_ground_truth: Dict[int, Set[int]] = (
        eval_positives_df.groupby("userId")["movieId"].apply(set).to_dict()
    )

    # Also extract movie genres mapping for content seed simulation
    movie_to_genres: Dict[int, List[str]] = {}
    for _, row in movies_df.iterrows():
        g_str = str(row["genres"])
        if g_str and g_str != "(no genres listed)":
            movie_to_genres[int(row["movieId"])] = [g.strip() for g in g_str.split("|") if g.strip()]

    # First interaction genre per user (simulating cold user profile / onboarding choice)
    user_first_genre: Dict[int, str] = {}
    sorted_cold_val = val_df[val_df["userId"].isin(eval_cold_users)].sort_values("timestamp")
    for _, row in sorted_cold_val.iterrows():
        u = int(row["userId"])
        if u not in user_first_genre:
            m = int(row["movieId"])
            if m in movie_to_genres and movie_to_genres[m]:
                user_first_genre[u] = movie_to_genres[m][0]

    # 3. Fit Fallback Models strictly on train
    logger.info("Fitting TimeDecayedPopularity from train interactions...")
    time_pop = TimeDecayedPopularity.fit_from_dataframe(
        train_df=train_df, movies_df=movies_df, half_life_days=730.0
    )
    pop_artifact_path = artifacts_dir / "fallback_popularity.json"
    time_pop.save(pop_artifact_path)

    logger.info("Fitting ContentBasedRecommender from catalog and train features...")
    content_recs = ContentBasedRecommender.fit_from_catalog(
        movies_df=movies_df, item_features_df=item_features_df
    )
    content_artifact_path = artifacts_dir / "fallback_content.json"
    content_recs.save(content_artifact_path)

    # 4. Generate Recommendations for Cold Users
    logger.info("Generating recommendations across 4 strategies for %d cold users...", len(eval_cold_users))
    catalog_ids = sorted(list(movies_df["movieId"].unique()))

    # Strategy 1: Random
    random.seed(42)
    random_recs: Dict[int, List[int]] = {
        u: random.sample(catalog_ids, 200) for u in eval_cold_users
    }

    # Strategy 2: Raw Training Popularity (top 200 items by train interaction count)
    raw_pop_counts = train_df["movieId"].value_counts().head(200).index.tolist()
    raw_pop_recs: Dict[int, List[int]] = {u: raw_pop_counts for u in eval_cold_users}

    # Strategy 3: Time-Decayed Popularity Fallback
    time_pop_top200 = time_pop.recommend_global(k=200)
    time_pop_recs: Dict[int, List[int]] = {u: time_pop_top200 for u in eval_cold_users}

    # Strategy 4: Content-Based Seed Fallback (memoized per genre)
    content_genre_recs: Dict[int, List[int]] = {}
    genre_recs_cache: Dict[str, List[int]] = {}
    for u in eval_cold_users:
        seed_genre = user_first_genre.get(u, "Drama")
        if seed_genre not in genre_recs_cache:
            genre_recs_cache[seed_genre] = content_recs.recommend_from_seed_genres(
                seed_genres=[seed_genre], k=200
            )
        content_genre_recs[u] = genre_recs_cache[seed_genre]

    # 5. Evaluate Metrics
    logger.info("Computing Recall@K and NDCG@K across methods on cold users...")
    k_list = [10, 50, 200]
    results = {
        "evaluation_cold_users_count": len(eval_cold_users),
        "random_baseline": evaluate_recommendations(random_recs, user_ground_truth, k_list=k_list),
        "global_popularity": evaluate_recommendations(raw_pop_recs, user_ground_truth, k_list=k_list),
        "time_decayed_popularity": evaluate_recommendations(time_pop_recs, user_ground_truth, k_list=k_list),
        "content_based_genre_seed": evaluate_recommendations(content_genre_recs, user_ground_truth, k_list=k_list),
    }

    # Log summary table
    logger.info("================ COLD-START EVALUATION RESULTS ================")
    logger.info("%-28s | %-10s | %-10s | %-10s | %-10s", "Strategy", "Recall@10", "NDCG@10", "Recall@50", "Recall@200")
    for name, key in [
        ("Random Baseline", "random_baseline"),
        ("Raw Global Popularity", "global_popularity"),
        ("Time-Decayed Popularity", "time_decayed_popularity"),
        ("Content-Based (Seed Genre)", "content_based_genre_seed"),
    ]:
        m = results[key]
        logger.info(
            "%-28s | %-10.5f | %-10.5f | %-10.5f | %-10.5f",
            name,
            m["recall@10"],
            m["ndcg@10"],
            m["recall@50"],
            m["recall@200"],
        )
    logger.info("===============================================================")

    # 6. Populate Redis Cache and Benchmark Latency
    cache = RedisCache(host="localhost", port=6379)
    cache_bench = {"redis_available": False, "p50_hit_ms": None, "p95_hit_ms": None}
    if cache.is_available():
        logger.info("Warming Redis cache with popularity lists and benchmark entries...")
        cache.set_global_popularity(time_pop.recommend_global(k=100))
        for genre in ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Thriller"]:
            cache.set_genre_popularity(genre, time_pop.recommend_by_genre(genre, k=100))

        # Benchmark cache hit latency on 500 requests
        sample_recs = [{"movieId": mid, "score": 1.0 / (idx + 1)} for idx, mid in enumerate(time_pop_top200[:10])]
        cache.set_recommendations(variant="v1", user_id=12345, k=10, recommendations=sample_recs)

        latencies_ms = []
        for _ in range(500):
            t0 = time.perf_counter()
            val = cache.get_recommendations(variant="v1", user_id=12345, k=10)
            t1 = time.perf_counter()
            assert val is not None
            latencies_ms.append((t1 - t0) * 1000.0)

        p50 = float(np.percentile(latencies_ms, 50))
        p95 = float(np.percentile(latencies_ms, 95))
        p99 = float(np.percentile(latencies_ms, 99))
        logger.info("Redis cache hit latency: p50 = %.3f ms, p95 = %.3f ms, p99 = %.3f ms", p50, p95, p99)
        cache_bench = {
            "redis_available": True,
            "p50_hit_ms": p50,
            "p95_hit_ms": p95,
            "p99_hit_ms": p99,
        }
    else:
        logger.warning("Redis is not available; cache benchmark skipped.")

    results["cache_benchmark"] = cache_bench

    # 7. Save Report
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Cold-start report saved to %s", report_path)


if __name__ == "__main__":
    main()
