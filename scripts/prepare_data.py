"""
Prepare Data Pipeline Script for Phase 1.
1. Loads and cleans MovieLens 25M interactions.
2. Performs strict temporal split (~80/10/10) with zero future leakage.
3. Fits ID vocabularies on training split only.
4. Generates a 1M interaction dev subset.
5. Saves Parquet splits and JSON vocabularies.
6. Evaluates Random and Popularity baselines on validation split.
7. Produces reports/split_report.json and reports/baselines.json.
"""

import json
import os
import sys

import numpy as np
import yaml

# Set up paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.reco.data.loader import clean_ratings, load_raw_ratings
from src.reco.data.split import compute_split_statistics, temporal_split
from src.reco.data.vocab import IdVocabulary
from src.reco.evaluation.baselines import PopularityBaseline, RandomBaseline
from src.reco.evaluation.metrics import evaluate_recommendations
from src.reco.utils.logging import get_logger
from src.reco.utils.seeds import set_seed

logger = get_logger("prepare_data")


def main() -> None:
    set_seed(42)

    config_path = os.path.join(PROJECT_ROOT, "configs", "data.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_dir = os.path.join(PROJECT_ROOT, config.get("raw_data_dir", "data/raw"))
    proc_dir = os.path.join(PROJECT_ROOT, config.get("processed_data_dir", "data/processed"))
    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(os.path.join(proc_dir, "dev"), exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Load and clean raw interactions
    raw_df = load_raw_ratings(raw_dir)
    clean_df = clean_ratings(raw_df, positive_threshold=float(config.get("positive_threshold", 4.0)))

    # 2. Temporal Split (80 / 10 / 10)
    train_ratio = float(config.get("train_ratio", 0.80))
    val_ratio = float(config.get("val_ratio", 0.10))
    test_ratio = float(config.get("test_ratio", 0.10))

    train_df, val_df, test_df = temporal_split(
        clean_df, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
    )

    # 3. Compute and save split statistics report
    split_stats = compute_split_statistics(train_df, val_df, test_df)
    split_report_path = os.path.join(reports_dir, "split_report.json")
    with open(split_report_path, "w", encoding="utf-8") as f:
        json.dump(split_stats, f, indent=2)
    logger.info("Saved split report to %s", split_report_path)

    # 4. Save full datasets as Parquet
    logger.info("Writing full Parquet splits...")
    train_df.to_parquet(os.path.join(proc_dir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(proc_dir, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(proc_dir, "test.parquet"), index=False)

    # 5. Build and save ID vocabularies (fit ONLY on train)
    logger.info("Building ID vocabularies from train split only...")
    user_vocab = IdVocabulary(name="user_id").fit(train_df["userId"].unique())
    movie_vocab = IdVocabulary(name="movie_id").fit(train_df["movieId"].unique())

    user_vocab.save(os.path.join(proc_dir, "vocab_users.json"))
    movie_vocab.save(os.path.join(proc_dir, "vocab_movies.json"))
    logger.info(
        "Train vocabularies: %d users, %d items.", len(user_vocab), len(movie_vocab)
    )

    # 6. Generate Dev Subset (1M interactions chronologically preserving temporal ratio)
    dev_size = int(config.get("dev_subset_size", 1000000))
    logger.info("Creating dev subset of %d interactions...", dev_size)
    dev_clean_df = clean_df.iloc[:dev_size].copy()
    dev_train, dev_val, dev_test = temporal_split(
        dev_clean_df, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio
    )
    dev_train.to_parquet(os.path.join(proc_dir, "dev", "train.parquet"), index=False)
    dev_val.to_parquet(os.path.join(proc_dir, "dev", "val.parquet"), index=False)
    dev_test.to_parquet(os.path.join(proc_dir, "dev", "test.parquet"), index=False)
    logger.info("Dev subset written to %s", os.path.join(proc_dir, "dev"))

    # 7. Evaluate Baselines on Validation Split (Positive Interactions)
    logger.info("Evaluating baselines on validation split...")
    val_positives = val_df[val_df["is_positive"] == 1]
    val_ground_truth = (
        val_positives.groupby("userId")["movieId"].apply(set).to_dict()
    )

    # Items already seen in training by each user
    train_seen_by_user = (
        train_df.groupby("userId")["movieId"].apply(set).to_dict()
    )

    # Evaluate on a representative sample of val users (up to 10,000 for fast reproducible evaluation)
    rng = np.random.default_rng(42)
    val_user_candidates = list(val_ground_truth.keys())
    if len(val_user_candidates) > 10000:
        eval_users = list(rng.choice(val_user_candidates, size=10000, replace=False))
    else:
        eval_users = val_user_candidates

    logger.info("Evaluating %d validation users with positive interactions.", len(eval_users))

    train_catalog = list(movie_vocab.id_to_index.keys())
    random_recommender = RandomBaseline(item_catalog=train_catalog, seed=42)
    pop_recommender = PopularityBaseline(train_df=train_df, positive_only=True)

    k_list = [10, 50, 200]
    random_recs = {}
    pop_recs = {}

    for u in eval_users:
        seen = train_seen_by_user.get(u, set())
        random_recs[u] = random_recommender.recommend(user_id=u, k=200, seen_items=seen)
        pop_recs[u] = pop_recommender.recommend(user_id=u, k=200, seen_items=seen)

    eval_gt = {u: val_ground_truth[u] for u in eval_users}
    random_metrics = evaluate_recommendations(random_recs, eval_gt, k_list=k_list)
    pop_metrics = evaluate_recommendations(pop_recs, eval_gt, k_list=k_list)

    baselines_report = {
        "evaluation_users_count": len(eval_users),
        "k_list": k_list,
        "random_baseline": random_metrics,
        "popularity_baseline": pop_metrics,
    }

    baselines_path = os.path.join(reports_dir, "baselines.json")
    with open(baselines_path, "w", encoding="utf-8") as f:
        json.dump(baselines_report, f, indent=2)

    logger.info("Saved baseline metrics to %s", baselines_path)
    logger.info("Baseline Results:")
    logger.info("Random:     %s", json.dumps(random_metrics, indent=2))
    logger.info("Popularity: %s", json.dumps(pop_metrics, indent=2))


if __name__ == "__main__":
    main()
