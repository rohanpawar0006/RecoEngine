"""
Feature Pipeline Execution.
Orchestrates computation and Parquet export of user and item features
strictly from the training partition.
"""

import os

from pyspark.sql import SparkSession

from src.reco.features.item_features import compute_item_features
from src.reco.features.spark_session import get_spark_session
from src.reco.features.user_features import compute_user_features
from src.reco.utils.logging import get_logger

logger = get_logger(__name__)


def build_and_save_features(
    train_parquet_path: str,
    movies_csv_path: str,
    output_dir: str,
    spark: SparkSession = None,
    last_n_items: int = 10,
) -> None:
    """
    Build user and item feature tables from training interactions and save as Parquet.

    Args:
        train_parquet_path: Path to the training interactions Parquet file.
        movies_csv_path: Path to movies.csv metadata.
        output_dir: Destination folder to write user_features.parquet and item_features.parquet.
        spark: Active SparkSession (creates a new one if None).
        last_n_items: Max number of recent item IDs in user history.
    """
    stop_spark_when_done = False
    if spark is None:
        spark = get_spark_session(app_name="FeaturePipeline")
        stop_spark_when_done = True

    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Reading training interactions from %s", train_parquet_path)
        train_df = spark.read.parquet(train_parquet_path)

        logger.info("Reading movie metadata from %s", movies_csv_path)
        movies_df = spark.read.csv(movies_csv_path, header=True, inferSchema=True)

        # 1. Compute and save item features
        item_features_df = compute_item_features(train_df=train_df, movies_df=movies_df)
        item_output_path = os.path.join(output_dir, "item_features.parquet")
        logger.info("Writing item features to %s ...", item_output_path)
        if os.path.isdir(item_output_path):
            import shutil
            shutil.rmtree(item_output_path)
        elif os.path.isfile(item_output_path):
            os.remove(item_output_path)
        item_pdf = item_features_df.toPandas()
        item_pdf.to_parquet(item_output_path, index=False)

        # 2. Compute and save user features
        user_features_df = compute_user_features(
            train_df=train_df, movies_df=movies_df, last_n=last_n_items
        )
        user_output_path = os.path.join(output_dir, "user_features.parquet")
        logger.info("Writing user features to %s ...", user_output_path)
        if os.path.isdir(user_output_path):
            import shutil
            shutil.rmtree(user_output_path)
        elif os.path.isfile(user_output_path):
            os.remove(user_output_path)
        user_pdf = user_features_df.toPandas()
        user_pdf.to_parquet(user_output_path, index=False)

        logger.info("Feature engineering complete: tables written to %s", output_dir)
    finally:
        if stop_spark_when_done:
            spark.stop()
