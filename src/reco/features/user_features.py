"""
User Feature Extraction using Apache Spark.
Computes user interaction counts, average ratings, positive counts,
recent item history (last N items), and top preferred genre strictly from the training window.
"""

import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.window import Window

from src.reco.utils.logging import get_logger

logger = get_logger(__name__)


def compute_user_features(
    train_df: DataFrame,
    movies_df: DataFrame,
    last_n: int = 10,
) -> DataFrame:
    """
    Compute user-level features using strictly the training partition.

    Args:
        train_df: Spark DataFrame of training interactions (userId, movieId, rating, timestamp, is_positive).
        movies_df: Spark DataFrame of movie metadata (movieId, title, genres).
        last_n: Number of most recent item interactions to keep in user history.

    Returns:
        Spark DataFrame of user features:
        - userId: int
        - user_interaction_count: int
        - user_avg_rating: float
        - user_positive_count: int
        - last_n_items: array<int>
        - top_genre: string
    """
    logger.info("Computing user features from training interactions (last_n=%d)...", last_n)

    # 1. Base aggregations: count, average rating, positive interaction count
    user_stats = train_df.groupBy("userId").agg(
        F.count("rating").alias("user_interaction_count"),
        F.round(F.avg("rating"), 4).alias("user_avg_rating"),
        F.sum(F.when(F.col("is_positive") == 1, 1).otherwise(0)).alias("user_positive_count"),
    )

    # 2. Last-N item history: window over userId ordered by timestamp descending
    history_window = Window.partitionBy("userId").orderBy(F.col("timestamp").desc())
    ranked_interactions = train_df.withColumn("rank", F.row_number().over(history_window))
    
    recent_interactions = ranked_interactions.filter(F.col("rank") <= last_n)
    # Collect items in chronological order for the history array
    user_history = (
        recent_interactions.groupBy("userId")
        .agg(F.collect_list("movieId").alias("last_n_items"))
    )

    # 3. Top genre affinity: based on positive interactions in training
    positives = train_df.filter(F.col("is_positive") == 1)
    user_movie_genres = (
        positives.join(movies_df.select("movieId", "genres"), on="movieId", how="inner")
        .withColumn("genre", F.explode(F.split(F.col("genres"), r"\|")))
        .filter(F.col("genre") != "(no genres listed)")
    )

    genre_counts = user_movie_genres.groupBy("userId", "genre").count()
    genre_window = Window.partitionBy("userId").orderBy(F.col("count").desc())
    
    top_genres = (
        genre_counts.withColumn("genre_rank", F.row_number().over(genre_window))
        .filter(F.col("genre_rank") == 1)
        .select("userId", F.col("genre").alias("top_genre"))
    )

    # 4. Join all user features together
    user_features = (
        user_stats.join(user_history, on="userId", how="left")
        .join(top_genres, on="userId", how="left")
        .withColumn("top_genre", F.coalesce(F.col("top_genre"), F.lit("Unknown")))
        .select(
            F.col("userId").cast("int"),
            F.col("user_interaction_count").cast("int"),
            F.col("user_avg_rating").cast("float"),
            F.col("user_positive_count").cast("int"),
            F.col("last_n_items"),
            F.col("top_genre"),
        )
    )

    return user_features
