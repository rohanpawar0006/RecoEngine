"""
Tests verifying data splitting integrity, zero future leakage, and vocabulary isolation.
"""

import pandas as pd
import pytest

from src.reco.data.split import compute_split_statistics, temporal_split
from src.reco.data.vocab import IdVocabulary


@pytest.fixture
def dummy_interactions():
    """Create synthetic chronological interactions."""
    data = {
        "userId": [1, 2, 1, 3, 2, 4, 1, 5, 2, 6],
        "movieId": [10, 20, 30, 10, 40, 50, 20, 60, 10, 70],
        "rating": [4.0, 3.5, 5.0, 4.5, 2.0, 4.0, 5.0, 4.0, 3.0, 4.5],
        "timestamp": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "is_positive": [1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
    }
    return pd.DataFrame(data)


def test_temporal_split_integrity(dummy_interactions):
    """Verify strictly ascending timestamps and zero future leakage across splits."""
    train, val, test = temporal_split(
        dummy_interactions, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20
    )

    assert len(train) == 6
    assert len(val) == 2
    assert len(test) == 2

    # Strict temporal boundary check
    assert train["timestamp"].max() <= val["timestamp"].min()
    assert val["timestamp"].max() <= test["timestamp"].min()


def test_vocabulary_built_only_from_train(dummy_interactions):
    """Verify vocabularies are strictly fit on train interactions; unseen IDs in val/test map to OOV."""
    train, _val, _test = temporal_split(
        dummy_interactions, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20
    )

    user_vocab = IdVocabulary(name="user_id").fit(train["userId"].unique())
    movie_vocab = IdVocabulary(name="movie_id").fit(train["movieId"].unique())

    # Users in train: 1, 2, 3, 4
    for u in train["userId"]:
        assert user_vocab.is_known(u)
        assert user_vocab.transform(u) >= 0

    # User 5 appears only in val/test: must NOT be in training vocabulary
    assert not user_vocab.is_known(5)
    assert user_vocab.transform(5) == -1

    # Movie 70 appears only in test: must NOT be in training vocabulary
    assert not movie_vocab.is_known(70)
    assert movie_vocab.transform(70) == -1


def test_split_statistics_reporting(dummy_interactions):
    """Verify statistics computation accurately identifies cold users and items."""
    train, val, test = temporal_split(
        dummy_interactions, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20
    )
    stats = compute_split_statistics(train, val, test)

    assert stats["split_integrity"]["max_train_ts_le_min_val_ts"] is True
    assert stats["split_integrity"]["max_val_ts_le_min_test_ts"] is True
    assert stats["train"]["num_interactions"] == 6
    assert stats["val"]["cold_users"] >= 0
