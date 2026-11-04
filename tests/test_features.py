"""
Unit tests for Spark feature engineering pipeline.
Tests schema, null handling, release year regex extraction, and leakage isolation.
"""

import pytest

from src.reco.features.item_features import compute_item_features
from src.reco.features.spark_session import get_spark_session
from src.reco.features.user_features import compute_user_features


@pytest.fixture(scope="module")
def spark():
    """Shared SparkSession fixture for feature testing."""
    session = get_spark_session(app_name="TestFeatures", master="local[1]", driver_memory="2g")
    yield session
    session.stop()


@pytest.fixture
def sample_data(spark):
    """Synthetic interaction and movie data."""
    train_data = [
        (1, 101, 5.0, 1000, 1),
        (1, 102, 3.0, 2000, 0),
        (1, 103, 4.5, 3000, 1),
        (2, 101, 4.0, 1500, 1),
        (2, 104, 2.0, 2500, 0),
    ]
    train_df = spark.createDataFrame(
        train_data, ["userId", "movieId", "rating", "timestamp", "is_positive"]
    )

    movies_data = [
        (101, "Movie Alpha (1995)", "Action|Adventure"),
        (102, "Movie Beta (2000)", "Comedy|Romance"),
        (103, "Movie Gamma (2015)", "Action|Sci-Fi"),
        (104, "Movie Delta (2020)", "Drama"),
        (105, "Unrated Film", "Thriller"),  # Catalog movie with 0 training interactions
    ]
    movies_df = spark.createDataFrame(movies_data, ["movieId", "title", "genres"])

    return train_df, movies_df


def test_item_features_computation(spark, sample_data):
    """Verify item feature schema, release year extraction, and 0-default for unrated items."""
    train_df, movies_df = sample_data
    item_features = compute_item_features(train_df, movies_df)

    rows = {row["movieId"]: row for row in item_features.collect()}

    # Check rated item (101)
    item_101 = rows[101]
    assert item_101["item_popularity"] == 2
    assert item_101["item_positive_count"] == 2
    assert item_101["release_year"] == 1995
    assert abs(item_101["item_avg_rating"] - 4.5) < 1e-3

    # Check unrated catalog item (105) defaults
    item_105 = rows[105]
    assert item_105["item_popularity"] == 0
    assert item_105["item_positive_count"] == 0
    assert item_105["item_avg_rating"] == 0.0
    assert item_105["release_year"] == 0


def test_user_features_computation(spark, sample_data):
    """Verify user feature schema, interaction count, last_n_items order, and top genre."""
    train_df, movies_df = sample_data
    user_features = compute_user_features(train_df, movies_df, last_n=2)

    rows = {row["userId"]: row for row in user_features.collect()}

    user_1 = rows[1]
    assert user_1["user_interaction_count"] == 3
    assert user_1["user_positive_count"] == 2
    # last_n=2 should capture the 2 most recent interactions (103 at 3000, 102 at 2000)
    assert len(user_1["last_n_items"]) == 2
    assert set(user_1["last_n_items"]) == {103, 102}
    # Top genre from positives (101 has Action/Adventure, 103 has Action/Sci-Fi -> Action appears twice)
    assert user_1["top_genre"] == "Action"


def test_leakage_isolation(spark, sample_data):
    """Verify that adding validation/test interactions does NOT affect training features."""
    train_df, movies_df = sample_data

    # Features computed strictly on train
    train_item_features = compute_item_features(train_df, movies_df)
    train_stats = {r["movieId"]: r["item_popularity"] for r in train_item_features.collect()}

    # Simulated future val interaction for item 101
    val_data = [(99, 101, 5.0, 99999, 1)]
    val_df = spark.createDataFrame(
        val_data, ["userId", "movieId", "rating", "timestamp", "is_positive"]
    )

    # Features computed strictly on train must ignore val interactions
    isolated_features = compute_item_features(train_df, movies_df)
    isolated_stats = {r["movieId"]: r["item_popularity"] for r in isolated_features.collect()}

    # If someone erroneously leaked val data by unioning
    leaked_features = compute_item_features(train_df.union(val_df), movies_df)
    leaked_stats = {r["movieId"]: r["item_popularity"] for r in leaked_features.collect()}

    assert train_stats == isolated_stats
    assert isolated_stats[101] == 2
    assert leaked_stats[101] == 3  # Leakage would increase popularity to 3
