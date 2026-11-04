"""
Data loader and cleaner for MovieLens 25M dataset.
Loads ratings, movies, and tags, removes duplicates, and marks implicit positives (rating >= 4.0).
"""

import os

import pandas as pd

from src.reco.utils.logging import get_logger

logger = get_logger(__name__)


def load_raw_ratings(raw_data_dir: str) -> pd.DataFrame:
    """Load ratings.csv from raw data directory."""
    path = os.path.join(raw_data_dir, "ml-25m", "ratings.csv")
    if not os.path.exists(path):
        # Fallback to direct path if unzipped without ml-25m wrapper
        path = os.path.join(raw_data_dir, "ratings.csv")
    logger.info("Loading ratings from %s ...", path)
    
    # Read with explicit optimized dtypes
    df = pd.read_csv(
        path,
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
    )
    logger.info("Loaded %d raw ratings.", len(df))
    return df


def load_raw_movies(raw_data_dir: str) -> pd.DataFrame:
    """Load movies.csv from raw data directory."""
    path = os.path.join(raw_data_dir, "ml-25m", "movies.csv")
    if not os.path.exists(path):
        path = os.path.join(raw_data_dir, "movies.csv")
    logger.info("Loading movies from %s ...", path)
    df = pd.read_csv(path, dtype={"movieId": "int32", "title": "string", "genres": "string"})
    return df


def clean_ratings(df: pd.DataFrame, positive_threshold: float = 4.0) -> pd.DataFrame:
    """
    Clean ratings data:
    1. Drop nulls / NaNs.
    2. Deduplicate (userId, movieId), keeping the latest interaction.
    3. Add binary 'is_positive' column where rating >= positive_threshold.
    4. Sort chronologically by timestamp.
    """
    initial_len = len(df)
    df = df.dropna(subset=["userId", "movieId", "rating", "timestamp"]).copy()
    
    # Deduplicate: if a user rated the same movie multiple times, keep latest
    df = df.sort_values("timestamp", ascending=True)
    df = df.drop_duplicates(subset=["userId", "movieId"], keep="last")
    
    deduped_len = len(df)
    if initial_len != deduped_len:
        logger.info("Removed %d duplicate/null interactions.", initial_len - deduped_len)
        
    df["is_positive"] = (df["rating"] >= positive_threshold).astype("int8")
    return df
