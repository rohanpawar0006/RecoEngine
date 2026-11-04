"""
Dataset preparation for Two-Tower Retrieval Model using TensorFlow.
Loads training interactions and creates optimized tf.data.Dataset pipelines.
"""


import numpy as np
import pandas as pd
import tensorflow as tf

from src.reco.data.vocab import IdVocabulary
from src.reco.utils.logging import get_logger

logger = get_logger(__name__)


def create_retrieval_dataset(
    interactions_path: str,
    user_vocab: IdVocabulary,
    movie_vocab: IdVocabulary,
    batch_size: int = 4096,
    positive_only: bool = True,
    max_samples: int | None = None,
    shuffle_buffer: int = 50000,
    seed: int = 42,
) -> tf.data.Dataset:
    """
    Create an optimized tf.data.Dataset of (userId_idx, movieId_idx) pairs for two-tower training.

    Args:
        interactions_path: Path to train.parquet file.
        user_vocab: IdVocabulary for users.
        movie_vocab: IdVocabulary for movies.
        batch_size: Mini-batch size.
        positive_only: If True, train only on positive interactions (is_positive == 1).
        max_samples: If set, use the most recent N interactions (recency weighting).
        shuffle_buffer: Number of elements to buffer for shuffling.
        seed: Random seed.

    Returns:
        tf.data.Dataset yielding batches of {"userId": tf.Tensor, "movieId": tf.Tensor}.
    """
    logger.info("Loading interactions from %s ...", interactions_path)
    df = pd.read_parquet(interactions_path, columns=["userId", "movieId", "is_positive", "timestamp"])

    if positive_only and "is_positive" in df.columns:
        df = df[df["is_positive"] == 1]

    if max_samples and len(df) > max_samples:
        df = df.sort_values("timestamp", ascending=True).iloc[-max_samples:]
        logger.info("Selected most recent %d positive interactions.", len(df))

    # Map raw IDs to contiguous integer indices in [0, V-1]
    user_indices = np.array([user_vocab.transform(u) for u in df["userId"].values], dtype=np.int32)
    movie_indices = np.array([movie_vocab.transform(m) for m in df["movieId"].values], dtype=np.int32)

    # Filter any OOV indices if present
    valid_mask = (user_indices >= 0) & (movie_indices >= 0)
    user_indices = user_indices[valid_mask]
    movie_indices = movie_indices[valid_mask]

    logger.info("Created training dataset with %d positive interaction pairs.", len(user_indices))

    dataset = tf.data.Dataset.from_tensor_slices({
        "userId": user_indices,
        "movieId": movie_indices,
    })

    if shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def get_catalog_indices(movie_vocab: IdVocabulary) -> np.ndarray:
    """Return array of all valid contiguous movie indices in the catalog."""
    return np.arange(len(movie_vocab), dtype=np.int32)
