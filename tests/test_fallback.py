"""
Unit tests for cold-start fallbacks: TimeDecayedPopularity and ContentBasedRecommender.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.reco.fallback.content_based import ContentBasedRecommender
from src.reco.fallback.popularity import TimeDecayedPopularity


@pytest.fixture
def synthetic_catalog():
    """Synthetic movies catalog."""
    return pd.DataFrame(
        [
            {"movieId": 1, "title": "SciFi Adventure (2015)", "genres": "Action|Adventure|Sci-Fi"},
            {"movieId": 2, "title": "Romantic Comedy (2010)", "genres": "Comedy|Romance"},
            {"movieId": 3, "title": "Space Odyssey (2016)", "genres": "Sci-Fi"},
            {"movieId": 4, "title": "Action Thriller (2014)", "genres": "Action|Thriller"},
            {"movieId": 5, "title": "Animation Fun (2018)", "genres": "Animation|Children"},
        ]
    )


@pytest.fixture
def synthetic_interactions():
    """Synthetic training interactions with timestamps."""
    return pd.DataFrame(
        [
            # Movie 3 has recent positive interactions (high time-decayed weight)
            {"userId": 1, "movieId": 3, "rating": 5.0, "timestamp": 1500000000},
            {"userId": 2, "movieId": 3, "rating": 4.5, "timestamp": 1490000000},
            # Movie 1 has older interactions
            {"userId": 1, "movieId": 1, "rating": 4.0, "timestamp": 1000000000},
            {"userId": 2, "movieId": 1, "rating": 4.0, "timestamp": 1100000000},
            {"userId": 3, "movieId": 1, "rating": 4.0, "timestamp": 1200000000},
            # Movie 2 has low rating
            {"userId": 1, "movieId": 2, "rating": 2.0, "timestamp": 1500000000},
            # Movie 4 has one interaction
            {"userId": 3, "movieId": 4, "rating": 4.5, "timestamp": 1480000000},
        ]
    )


def test_time_decayed_popularity(synthetic_interactions, synthetic_catalog):
    """Verify time-decayed scoring ranks recent high-rated items higher than old ones."""
    recommender = TimeDecayedPopularity.fit_from_dataframe(
        synthetic_interactions, synthetic_catalog, half_life_days=365.0
    )

    # Movie 3 should be near the top due to recent 5.0 and 4.5 ratings despite fewer total counts
    top_global = recommender.recommend_global(k=3)
    assert len(top_global) == 3
    assert 3 in top_global

    # Test seen item filtering
    filtered = recommender.recommend_global(k=3, seen_items={3})
    assert 3 not in filtered

    # Test genre filtering
    scifi_top = recommender.recommend_by_genre(genre="Sci-Fi", k=2)
    assert 3 in scifi_top or 1 in scifi_top


def test_time_decayed_popularity_save_load(tmp_path: Path, synthetic_interactions, synthetic_catalog):
    """Test serialization and deserialization of TimeDecayedPopularity."""
    recommender = TimeDecayedPopularity.fit_from_dataframe(
        synthetic_interactions, synthetic_catalog
    )
    save_path = tmp_path / "pop.json"
    recommender.save(save_path)

    loaded = TimeDecayedPopularity.load(save_path)
    assert loaded.recommend_global(k=2) == recommender.recommend_global(k=2)
    assert loaded.recommend_by_genre("Sci-Fi", k=2) == recommender.recommend_by_genre("Sci-Fi", k=2)


def test_content_based_recommendations(synthetic_catalog):
    """Verify content-based recommendations from seed genres and seed items."""
    recommender = ContentBasedRecommender.fit_from_catalog(synthetic_catalog)

    # 1. Seed genre: Sci-Fi should rank movie 3 and movie 1 highest
    recs = recommender.recommend_from_seed_genres(seed_genres=["Sci-Fi"], k=2)
    assert len(recs) == 2
    assert set(recs) == {1, 3}

    # 2. Seed genre with seen filter: if 3 is seen, 1 is returned
    recs_seen = recommender.recommend_from_seed_genres(seed_genres=["Sci-Fi"], k=1, seen_items={3})
    assert recs_seen == [1]

    # 3. Seed items: seed movie 2 (Comedy|Romance)
    recs_items = recommender.recommend_from_seed_items(seed_movie_ids=[2], k=2)
    assert 2 not in recs_items  # seed items automatically excluded
    assert len(recs_items) <= 2

    # 4. Unknown genre fallback
    fallback_recs = recommender.recommend_from_seed_genres(seed_genres=["NonExistentGenre"], k=2)
    assert len(fallback_recs) == 2


def test_content_based_save_load(tmp_path: Path, synthetic_catalog):
    """Verify save and load roundtrip of ContentBasedRecommender."""
    recommender = ContentBasedRecommender.fit_from_catalog(synthetic_catalog)
    save_path = tmp_path / "content.json"
    recommender.save(save_path)

    loaded = ContentBasedRecommender.load(save_path)
    assert loaded.recommend_from_seed_genres(["Animation"], k=1) == [5]
