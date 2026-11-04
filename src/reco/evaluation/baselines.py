"""
Baseline recommenders: Random and Global Popularity.
Rules:
- Popularity must be computed strictly from the training split.
- Seeds must be fixed for reproducibility.
"""

import numpy as np
import pandas as pd

from src.reco.utils.logging import get_logger

logger = get_logger(__name__)


class RandomBaseline:
    """Random recommendation baseline: samples items uniformly from the train catalog."""

    def __init__(self, item_catalog: list[int], seed: int = 42) -> None:
        self.item_catalog = np.array(sorted(set(item_catalog)))
        self.rng = np.random.default_rng(seed)

    def recommend(self, user_id: int, k: int = 10, seen_items: set[int] | None = None) -> list[int]:
        """Sample k random items, optionally excluding seen items."""
        if seen_items:
            candidates = np.array([item for item in self.item_catalog if item not in seen_items])
        else:
            candidates = self.item_catalog

        if len(candidates) == 0:
            return []

        k = min(k, len(candidates))
        return list(self.rng.choice(candidates, size=k, replace=False))


class PopularityBaseline:
    """Popularity recommendation baseline: ranks items by training interaction frequency."""

    def __init__(self, train_df: pd.DataFrame, positive_only: bool = True) -> None:
        """
        Fit popularity strictly on training interactions.

        Args:
            train_df: Training interactions DataFrame.
            positive_only: If True, count only positive interactions (rating >= 4.0).
        """
        if positive_only and "is_positive" in train_df.columns:
            subset = train_df[train_df["is_positive"] == 1]
        else:
            subset = train_df

        pop_counts = subset["movieId"].value_counts()
        self.popular_items: list[int] = pop_counts.index.tolist()
        logger.info(
            "Popularity baseline fit on %d interactions across %d unique items.",
            len(subset),
            len(self.popular_items),
        )

    def recommend(self, user_id: int, k: int = 10, seen_items: set[int] | None = None) -> list[int]:
        """Return top-k popular items, optionally filtering items already seen in train."""
        if not seen_items:
            return self.popular_items[:k]

        recs: list[int] = []
        for item in self.popular_items:
            if item not in seen_items:
                recs.append(item)
                if len(recs) == k:
                    break
        return recs
