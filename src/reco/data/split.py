"""
Temporal train / validation / test splitter.
Strict rule: Split is strictly by timestamp quantile.
Zero future leakage: No information from val/test may leak into train.
"""

from typing import Any

import numpy as np
import pandas as pd

from src.reco.utils.logging import get_logger

logger = get_logger(__name__)


def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame strictly by timestamp into train, val, and test partitions.

    Args:
        df: Input DataFrame with 'timestamp' column.
        train_ratio: Fraction of interactions for training (default: 0.80).
        val_ratio: Fraction of interactions for validation (default: 0.10).
        test_ratio: Fraction of interactions for testing (default: 0.10).

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"
    
    # Sort chronologically
    df_sorted = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_cutoff = df_sorted.iloc[train_end]["timestamp"]
    val_cutoff = df_sorted.iloc[val_end]["timestamp"]

    train_df = df_sorted[df_sorted["timestamp"] < train_cutoff].copy()
    val_df = df_sorted[(df_sorted["timestamp"] >= train_cutoff) & (df_sorted["timestamp"] < val_cutoff)].copy()
    test_df = df_sorted[df_sorted["timestamp"] >= val_cutoff].copy()

    # Integrity assertions: max train timestamp <= min val timestamp <= min test timestamp
    max_train_ts = train_df["timestamp"].max()
    min_val_ts = val_df["timestamp"].min()
    max_val_ts = val_df["timestamp"].max()
    min_test_ts = test_df["timestamp"].min()

    assert max_train_ts <= min_val_ts, (
        f"Data leakage detected! max_train_ts ({max_train_ts}) > min_val_ts ({min_val_ts})"
    )
    assert max_val_ts <= min_test_ts, (
        f"Data leakage detected! max_val_ts ({max_val_ts}) > min_test_ts ({min_test_ts})"
    )

    logger.info(
        "Temporal split complete: Train=%d (%s to %s), Val=%d (%s to %s), Test=%d (%s to %s)",
        len(train_df),
        pd.to_datetime(train_df["timestamp"].min(), unit="s"),
        pd.to_datetime(max_train_ts, unit="s"),
        len(val_df),
        pd.to_datetime(min_val_ts, unit="s"),
        pd.to_datetime(max_val_ts, unit="s"),
        len(test_df),
        pd.to_datetime(min_test_ts, unit="s"),
        pd.to_datetime(test_df["timestamp"].max(), unit="s"),
    )

    return train_df, val_df, test_df


def compute_split_statistics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute detailed audit statistics for train, validation, and test splits."""
    train_users = set(train_df["userId"].unique())
    train_items = set(train_df["movieId"].unique())

    val_users = set(val_df["userId"].unique())
    val_items = set(val_df["movieId"].unique())

    test_users = set(test_df["userId"].unique())
    test_items = set(test_df["movieId"].unique())

    val_cold_users = val_users - train_users
    val_cold_items = val_items - train_items

    test_cold_users = test_users - train_users
    test_cold_items = test_items - train_items

    stats = {
        "train": {
            "num_interactions": len(train_df),
            "num_positive_interactions": int(train_df["is_positive"].sum()),
            "num_users": len(train_users),
            "num_items": len(train_items),
            "min_timestamp": int(train_df["timestamp"].min()),
            "max_timestamp": int(train_df["timestamp"].max()),
            "start_date": str(pd.to_datetime(train_df["timestamp"].min(), unit="s")),
            "end_date": str(pd.to_datetime(train_df["timestamp"].max(), unit="s")),
        },
        "val": {
            "num_interactions": len(val_df),
            "num_positive_interactions": int(val_df["is_positive"].sum()),
            "num_users": len(val_users),
            "num_items": len(val_items),
            "cold_users": len(val_cold_users),
            "cold_user_pct": float(len(val_cold_users) / len(val_users)) if val_users else 0.0,
            "cold_items": len(val_cold_items),
            "cold_item_pct": float(len(val_cold_items) / len(val_items)) if val_items else 0.0,
            "min_timestamp": int(val_df["timestamp"].min()),
            "max_timestamp": int(val_df["timestamp"].max()),
            "start_date": str(pd.to_datetime(val_df["timestamp"].min(), unit="s")),
            "end_date": str(pd.to_datetime(val_df["timestamp"].max(), unit="s")),
        },
        "test": {
            "num_interactions": len(test_df),
            "num_positive_interactions": int(test_df["is_positive"].sum()),
            "num_users": len(test_users),
            "num_items": len(test_items),
            "cold_users": len(test_cold_users),
            "cold_user_pct": float(len(test_cold_users) / len(test_users)) if test_users else 0.0,
            "cold_items": len(test_cold_items),
            "cold_item_pct": float(len(test_cold_items) / len(test_items)) if test_items else 0.0,
            "min_timestamp": int(test_df["timestamp"].min()),
            "max_timestamp": int(test_df["timestamp"].max()),
            "start_date": str(pd.to_datetime(test_df["timestamp"].min(), unit="s")),
            "end_date": str(pd.to_datetime(test_df["timestamp"].max(), unit="s")),
        },
        "split_integrity": {
            "max_train_ts_le_min_val_ts": bool(train_df["timestamp"].max() <= val_df["timestamp"].min()),
            "max_val_ts_le_min_test_ts": bool(val_df["timestamp"].max() <= test_df["timestamp"].min()),
        },
    }
    return stats
