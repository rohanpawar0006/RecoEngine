"""
Item Feature Extraction using Apache Spark.
Computes item popularity, average rating, positive counts from the training window,
and extracts release year and genres from movie metadata.
"""

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from src.reco.utils.logging import get_logger

logger = get_logger(__name__)


def compute_item_features(
    train_df: DataFrame,
    movies_df: DataFrame,
) -> DataFrame:
    """
    Compute item-level features using strictly the training partition.

    Args:
        train_df: Spark DataFrame of training interactions (userId, movieId, rating, timestamp, is_positive).
        movies_df: Spark DataFrame of movie metadata (movieId, title, genres).

    Returns:
        Spark DataFrame of item features:
        - movieId: int
        - title: string
        - genres: string
        - release_year: int
        - item_popularity: int
        - item_avg_rating: float
        - item_positive_count: int
    """
    logger.info("Computing item features from training interactions...")

    # Aggregate statistics strictly from training interactions
    item_stats = train_df.groupBy("movieId").agg(
        F.count("rating").alias("item_popularity"),
        F.round(F.avg("rating"), 4).alias("item_avg_rating"),
        F.sum(F.when(F.col("is_positive") == 1, 1).otherwise(0)).alias("item_positive_count"),
    )

    # Extract 4-digit release year from title e.g. "Toy Story (1995)"
    # Regex returns empty string if no match; handle safely without ANSI cast error
    year_str = F.regexp_extract(F.col("title"), r"\((\d{4})\)$", 1)
    movies_with_year = movies_df.withColumn(
        "release_year",
        F.when(year_str != "", year_str.cast("int")).otherwise(F.lit(0)),
    )

    # Join metadata with training stats; catalog items with no training interactions get 0 defaults
    item_features = (
        movies_with_year.join(item_stats, on="movieId", how="left")
        .withColumn("item_popularity", F.coalesce(F.col("item_popularity"), F.lit(0)).cast("int"))
        .withColumn("item_avg_rating", F.coalesce(F.col("item_avg_rating"), F.lit(0.0)).cast("float"))
        .withColumn("item_positive_count", F.coalesce(F.col("item_positive_count"), F.lit(0)).cast("int"))
        .select(
            "movieId",
            "title",
            "genres",
            "release_year",
            "item_popularity",
            "item_avg_rating",
            "item_positive_count",
        )
    )

    return item_features
