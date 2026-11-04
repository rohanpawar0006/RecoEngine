"""
Content-based recommender for cold-start users with seed preferences.
Matches users based on seed genres or seed movie likes using genre representations
and popularity-smoothed quality weighting.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)

ALL_GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "IMAX",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


class ContentBasedRecommender:
    """
    Content-based recommender leveraging movie genres and catalog quality statistics.
    Computes cosine similarity between user profile (seed genres or seed movie likes)
    and candidate items in the catalog.
    """

    def __init__(
        self,
        movie_metadata: Dict[int, Dict[str, any]],
        genre_to_idx: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            movie_metadata: Dict mapping movieId -> {
                "title": str,
                "genres": list[str],
                "genre_vector": list[float],
                "norm": float,
                "popularity": float,
                "avg_rating": float,
            }
            genre_to_idx: Mapping from genre name to vector index.
        """
        self.movie_metadata = movie_metadata
        self.genre_to_idx = genre_to_idx or {g: i for i, g in enumerate(ALL_GENRES)}

    @classmethod
    def fit_from_catalog(
        cls,
        movies_df: pd.DataFrame,
        item_features_df: Optional[pd.DataFrame] = None,
    ) -> ContentBasedRecommender:
        """
        Fit content-based index from catalog movies and training item features.
        Item features provide popularity and average rating derived strictly from train.
        """
        genre_to_idx = {g: i for i, g in enumerate(ALL_GENRES)}
        num_genres = len(genre_to_idx)

        # Merge with item features if available
        features_dict: Dict[int, Dict[str, float]] = {}
        if item_features_df is not None:
            for _, row in item_features_df.iterrows():
                features_dict[int(row["movieId"])] = {
                    "popularity": float(row.get("item_popularity", 0)),
                    "avg_rating": float(row.get("item_avg_rating", 0.0)),
                }

        metadata: Dict[int, Dict[str, any]] = {}
        for _, row in movies_df.iterrows():
            m_id = int(row["movieId"])
            title = str(row["title"])
            genres_raw = str(row["genres"])

            genres = [
                g.strip()
                for g in genres_raw.split("|")
                if g.strip() and g.strip() != "(no genres listed)"
            ]
            vec = [0.0] * num_genres
            for g in genres:
                if g in genre_to_idx:
                    vec[genre_to_idx[g]] = 1.0

            norm = math.sqrt(sum(v * v for v in vec))
            feat = features_dict.get(m_id, {"popularity": 0.0, "avg_rating": 0.0})

            metadata[m_id] = {
                "title": title,
                "genres": genres,
                "genre_vector": vec,
                "norm": norm if norm > 0 else 1.0,
                "popularity": feat["popularity"],
                "avg_rating": feat["avg_rating"],
            }

        logger.info("ContentBasedRecommender indexed %d catalog items.", len(metadata))
        return cls(movie_metadata=metadata, genre_to_idx=genre_to_idx)

    def recommend_from_seed_genres(
        self,
        seed_genres: List[str],
        k: int = 10,
        seen_items: Optional[Set[int]] = None,
    ) -> List[int]:
        """
        Recommend items matching a user's chosen seed genres.
        Ranks by cosine similarity to seed profile multiplied by log quality weight.
        """
        seen = seen_items or set()
        user_vec = [0.0] * len(self.genre_to_idx)
        for g in seed_genres:
            if g in self.genre_to_idx:
                user_vec[self.genre_to_idx[g]] = 1.0

        user_norm = math.sqrt(sum(v * v for v in user_vec))
        if user_norm == 0.0:
            # Fallback if unknown genres provided: rank purely by popularity * rating
            return self._fallback_by_popularity(k=k, seen=seen)

        scores: List[tuple[int, float]] = []
        for m_id, meta in self.movie_metadata.items():
            if m_id in seen:
                continue

            # Dot product
            dot = sum(u * v for u, v in zip(user_vec, meta["genre_vector"]))
            if dot <= 0.0:
                continue

            cos_sim = dot / (user_norm * meta["norm"])
            # Quality prior: log(1 + popularity) * rating factor
            quality_factor = math.log1p(max(meta["popularity"], 0.0)) * (
                (meta["avg_rating"] / 5.0) if meta["avg_rating"] > 0 else 0.5
            )
            final_score = cos_sim * (1.0 + 0.2 * quality_factor)
            scores.append((m_id, final_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = [m_id for m_id, _ in scores[:k]]

        # Pad with general popularity if fewer than k items match
        if len(results) < k:
            results.extend(self._fallback_by_popularity(k=k - len(results), seen=seen | set(results)))
        return results

    def recommend_from_seed_items(
        self,
        seed_movie_ids: List[int],
        k: int = 10,
        seen_items: Optional[Set[int]] = None,
    ) -> List[int]:
        """
        Recommend items similar to seed movie IDs liked by the user.
        Averages seed movies' genre vectors and ranks other catalog items by similarity.
        """
        seen = (seen_items or set()) | set(seed_movie_ids)
        user_vec = [0.0] * len(self.genre_to_idx)
        valid_seeds = 0

        for s_id in seed_movie_ids:
            if s_id in self.movie_metadata:
                for i, v in enumerate(self.movie_metadata[s_id]["genre_vector"]):
                    user_vec[i] += v
                valid_seeds += 1

        if valid_seeds == 0:
            return self._fallback_by_popularity(k=k, seen=seen)

        # Average vector
        user_vec = [v / valid_seeds for v in user_vec]
        user_norm = math.sqrt(sum(v * v for v in user_vec))
        if user_norm == 0.0:
            return self._fallback_by_popularity(k=k, seen=seen)

        scores: List[tuple[int, float]] = []
        for m_id, meta in self.movie_metadata.items():
            if m_id in seen:
                continue

            dot = sum(u * v for u, v in zip(user_vec, meta["genre_vector"]))
            if dot <= 0.0:
                continue

            cos_sim = dot / (user_norm * meta["norm"])
            quality_factor = math.log1p(max(meta["popularity"], 0.0)) * (
                (meta["avg_rating"] / 5.0) if meta["avg_rating"] > 0 else 0.5
            )
            final_score = cos_sim * (1.0 + 0.2 * quality_factor)
            scores.append((m_id, final_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = [m_id for m_id, _ in scores[:k]]
        if len(results) < k:
            results.extend(self._fallback_by_popularity(k=k - len(results), seen=seen | set(results)))
        return results

    def _fallback_by_popularity(self, k: int, seen: Set[int]) -> List[int]:
        """Return top items by popularity and rating when no matches exist."""
        ranked = sorted(
            [
                (m_id, meta["popularity"] * (meta["avg_rating"] if meta["avg_rating"] > 0 else 1.0))
                for m_id, meta in self.movie_metadata.items()
                if m_id not in seen
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        return [m_id for m_id, _ in ranked[:k]]

    def save(self, filepath: str | Path) -> None:
        """Serialize metadata to JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "genre_to_idx": self.genre_to_idx,
            "movie_metadata": self.movie_metadata,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.info("Saved ContentBasedRecommender to %s", path)

    @classmethod
    def load(cls, filepath: str | Path) -> ContentBasedRecommender:
        """Load metadata from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        metadata = {int(k): v for k, v in data["movie_metadata"].items()}
        return cls(movie_metadata=metadata, genre_to_idx=data["genre_to_idx"])

