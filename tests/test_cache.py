"""
Unit tests for Redis cache layer: key versioning, TTL, and graceful degradation.
"""

import pytest

from src.reco.serving.cache import RedisCache


def test_key_formatting():
    """Verify standardized key schema from Architecture.md."""
    key = RedisCache.format_rec_key(variant="v1", user_id=42, k=10)
    assert key == "rec:v1:42:10"

    key_b = RedisCache.format_rec_key(variant="B", user_id=1001, k=20)
    assert key_b == "rec:B:1001:20"


def test_graceful_degradation_when_redis_unreachable():
    """Verify that when Redis is unreachable or disabled, cache returns None / False without raising."""
    # 1. Disabled mode
    disabled_cache = RedisCache(enabled=False)
    assert disabled_cache.is_available() is False
    assert disabled_cache.get_recommendations(variant="v1", user_id=999, k=10) is None
    assert (
        disabled_cache.set_recommendations(
            variant="v1", user_id=999, k=10, recommendations=[{"item_id": 1, "score": 0.9}]
        )
        is False
    )
    assert disabled_cache.get_global_popularity(k=10) is None
    assert disabled_cache.set_global_popularity(movie_ids=[1, 2, 3]) is False
    assert disabled_cache.get_user_features(user_id=999) is None
    assert disabled_cache.set_user_features(user_id=999, features={"count": 5}) is False
    assert disabled_cache.flush_variant("v1") == 0

    # 2. Unreachable port with short timeout
    unreachable_cache = RedisCache(host="127.0.0.1", port=65432, socket_timeout=0.05)
    assert unreachable_cache.is_available() is False
    assert unreachable_cache.get_recommendations(variant="v1", user_id=999, k=10) is None
    assert (
        unreachable_cache.set_recommendations(
            variant="v1", user_id=999, k=10, recommendations=[{"item_id": 1, "score": 0.9}]
        )
        is False
    )


def test_live_redis_cache_operations():
    """Verify live Redis operations if local Redis server is active."""
    cache = RedisCache(host="localhost", port=6379, socket_timeout=1.0)
    if not cache.is_available():
        pytest.skip("Local Redis server not active; skipping live integration tests.")

    test_variant = "test_v1"
    test_user = 88888
    test_k = 5
    sample_recs = [
        {"movieId": 101, "score": 0.95, "title": "Test Alpha"},
        {"movieId": 102, "score": 0.88, "title": "Test Beta"},
    ]

    # Set and get recommendations with TTL
    success = cache.set_recommendations(
        variant=test_variant, user_id=test_user, k=test_k, recommendations=sample_recs, ttl=60
    )
    assert success is True

    cached = cache.get_recommendations(variant=test_variant, user_id=test_user, k=test_k)
    assert cached == sample_recs

    # Verify TTL
    rec_key = cache.format_rec_key(test_variant, test_user, test_k)
    ttl = cache.get_ttl(rec_key)
    assert 0 < ttl <= 60

    # Set and get popularity
    pop_ok = cache.set_global_popularity(movie_ids=[101, 102, 103], ttl=60)
    assert pop_ok is True
    cached_pop = cache.get_global_popularity(k=2)
    assert cached_pop == [101, 102]

    # Set and get genre popularity
    genre_ok = cache.set_genre_popularity(genre="Sci-Fi", movie_ids=[103, 104], ttl=60)
    assert genre_ok is True
    cached_genre = cache.get_genre_popularity(genre="Sci-Fi", k=5)
    assert cached_genre == [103, 104]

    # Set and get user features
    feat_ok = cache.set_user_features(user_id=test_user, features={"top_genre": "Action"}, ttl=60)
    assert feat_ok is True
    cached_feat = cache.get_user_features(user_id=test_user)
    assert cached_feat == {"top_genre": "Action"}

    # Test flush_variant deletes variant recs but leaves popularity intact
    deleted_count = cache.flush_variant(test_variant)
    assert deleted_count >= 1
    assert cache.get_recommendations(variant=test_variant, user_id=test_user, k=test_k) is None
    assert cache.get_global_popularity(k=2) == [101, 102]
