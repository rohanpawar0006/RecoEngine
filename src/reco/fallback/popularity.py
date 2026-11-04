"""
Time-decayed popularity recommender for cold-start users and global fallbacks.
Computes global and per-genre rankings strictly from training interactions.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class TimeDecayedPopularity:
    """
    Time-decayed popularity recommender.
    Assigns each interaction a weight: w = exp(-lambda * (t_max - t)),
    where lambda = ln(2) / half_life_seconds.
    """

    def __init__(
        self,
        global_rankings: Optional[List[Tuple[int, float]]] = None,
        genre_rankings: Optional[Dict[str, List[Tuple[int, float]]]] = None,
    ):
        """
        Args:
            global_rankings: List of (movieId, score) tuples in descending order.
            genre_rankings: Mapping of genre -> List of (movieId, score) tuples.
        """
        self.global_rankings: List[Tuple[int, float]] = global_rankings or []
        self.genre_rankings: Dict[str, List[Tuple[int, float]]] = genre_rankings or {}

    @classmethod
    def fit_from_dataframe(
        cls,
        train_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        half_life_days: float = 730.0,  # 2 years half-life
    ) -> TimeDecayedPopularity:
        """
        Compute time-decayed scores from training interactions and movies catalog.
        Strictly training interactions only (zero leakage).
        """
        logger.info("Computing time-decayed popularity from %d training rows...", len(train_df))
        max_ts = float(train_df["timestamp"].max())
        half_life_seconds = half_life_days * 86400.0
        decay_lambda = math.log(2.0) / max(half_life_seconds, 1.0)

        # Vectorized time-decay weight computation
        time_diff = (max_ts - train_df["timestamp"].astype(float)).clip(lower=0.0)
        weights = (-decay_lambda * time_diff).apply(math.exp)

        # Scale weights by rating (or positive indicator: rating >= 4.0 gets full weight, else half)
        rating_mult = (train_df["rating"] >= 4.0).astype(float) * 0.5 + 0.5
        total_weights = weights * rating_mult

        movie_scores = total_weights.groupby(train_df["movieId"]).sum().to_dict()

        # Merge with catalog to get genres and ensure catalog coverage
        catalog_movies = movies_df[["movieId", "genres"]].drop_duplicates("movieId")
        
        # Build global rankings
        sorted_global = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)

        # Build per-genre rankings
        genre_to_scores: Dict[str, List[Tuple[int, float]]] = {}
        for _, row in catalog_movies.iterrows():
            m_id = int(row["movieId"])
            if m_id not in movie_scores:
                continue
            score = float(movie_scores[m_id])
            genres_str = str(row["genres"])
            if genres_str and genres_str != "(no genres listed)":
                for genre in genres_str.split("|"):
                    genre = genre.strip()
                    if genre:
                        if genre not in genre_to_scores:
                            genre_to_scores[genre] = []
                        genre_to_scores[genre].append((m_id, score))

        # Sort each genre list descending
        for genre in genre_to_scores:
            genre_to_scores[genre].sort(key=lambda x: x[1], reverse=True)

        logger.info(
            "Time-decayed popularity fit complete. Global: %d items, Genres: %d",
            len(sorted_global),
            len(genre_to_scores),
        )
        return cls(global_rankings=sorted_global, genre_rankings=genre_to_scores)

    def recommend_global(
        self,
        k: int = 10,
        seen_items: Optional[Set[int]] = None,
    ) -> List[int]:
        """Return top-k globally popular items excluding seen_items."""
        seen = seen_items or set()
        results: List[int] = []
        for movie_id, _ in self.global_rankings:
            if movie_id not in seen:
                results.append(movie_id)
                if len(results) >= k:
                    break
        return results

    def recommend_by_genre(
        self,
        genre: str,
        k: int = 10,
        seen_items: Optional[Set[int]] = None,
    ) -> List[int]:
        """Return top-k popular items for a specific genre excluding seen_items."""
        seen = seen_items or set()
        genre_list = self.genre_rankings.get(genre, [])
        results: List[int] = []
        for movie_id, _ in genre_list:
            if movie_id not in seen:
                results.append(movie_id)
                if len(results) >= k:
                    break
        # Fall back to global if genre has fewer than k items
        if len(results) < k:
            seen_with_results = seen | set(results)
            results.extend(self.recommend_global(k=k - len(results), seen_items=seen_with_results))
        return results

    def save(self, filepath: str | Path) -> None:
        """Serialize rankings to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "global_rankings": self.global_rankings[:1000],  # Keep top 1000 for compact storage
            "genre_rankings": {
                g: items[:200] for g, items in self.genre_rankings.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.info("Saved TimeDecayedPopularity to %s", path)

    @classmethod
    def load(cls, filepath: str | Path) -> TimeDecayedPopularity:
        """Load rankings from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        global_rankings = [tuple(x) for x in data["global_rankings"]]
        genre_rankings = {
            g: [tuple(x) for x in items]
            for g, items in data["genre_rankings"].items()
        }
        return cls(global_rankings=global_rankings, genre_rankings=genre_rankings)

