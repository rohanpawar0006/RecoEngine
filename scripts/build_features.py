"""
Build Features Script for Phase 2.
Executes the Spark feature pipeline for both full training interactions and dev subset.
Produces:
- data/features/item_features.parquet
- data/features/user_features.parquet
- data/features/dev/item_features.parquet
- data/features/dev/user_features.parquet
"""

import os
import sys

import yaml

# Set up project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.reco.features.pipeline import build_and_save_features
from src.reco.features.spark_session import get_spark_session
from src.reco.utils.logging import get_logger

logger = get_logger("build_features")


def main() -> None:
    config_path = os.path.join(PROJECT_ROOT, "configs", "data.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    proc_dir = os.path.join(PROJECT_ROOT, config.get("processed_data_dir", "data/processed"))
    feat_dir = os.path.join(PROJECT_ROOT, config.get("features_data_dir", "data/features"))
    raw_dir = os.path.join(PROJECT_ROOT, config.get("raw_data_dir", "data/raw"))

    movies_csv_path = os.path.join(raw_dir, "ml-25m", "movies.csv")
    if not os.path.exists(movies_csv_path):
        movies_csv_path = os.path.join(raw_dir, "movies.csv")

    full_train_parquet = os.path.join(proc_dir, "train.parquet")
    dev_train_parquet = os.path.join(proc_dir, "dev", "train.parquet")

    # Start single SparkSession for building both feature sets
    spark = get_spark_session(app_name="BuildAllFeatures", driver_memory="4g")

    try:
        # 1. Build features for dev subset
        logger.info("=== Building Features for Dev Subset ===")
        dev_feat_dir = os.path.join(feat_dir, "dev")
        build_and_save_features(
            train_parquet_path=dev_train_parquet,
            movies_csv_path=movies_csv_path,
            output_dir=dev_feat_dir,
            spark=spark,
            last_n_items=10,
        )

        # 2. Build features for full dataset
        logger.info("=== Building Features for Full Dataset ===")
        build_and_save_features(
            train_parquet_path=full_train_parquet,
            movies_csv_path=movies_csv_path,
            output_dir=feat_dir,
            spark=spark,
            last_n_items=10,
        )

        logger.info("=== Feature Building Successfully Completed ===")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
