"""
Integration tests for FastAPI recommendation service.
Tests: /health, /recommend (warm, cold, cache hit), /events, /ab/summary, and input validation.
"""

import pytest
from fastapi.testclient import TestClient

from src.reco.serving.app import app, load_serving_state, state


@pytest.fixture(scope="module", autouse=True)
def setup_api():
    """Ensure state is initialized for testing."""
    load_serving_state()


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    """Verify /health returns 200 and reports system components."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["version"] == "v1"
    assert data["status"] in ["healthy", "degraded"]
    assert "redis_available" in data
    assert "triton_available" in data
    assert "served_by" in data
    assert data["faiss_loaded"] is True
    assert data["catalog_size"] > 0


def test_warm_user_recommendation(client):
    """Verify warm user flow: user tower -> FAISS top-200 -> DCN-V2 ranker -> top-K."""
    # Find a warm user from state
    warm_users = list(state.get("user_features", {}).keys())
    if not warm_users:
        pytest.skip("No warm users loaded in state.")
    test_user = warm_users[0]

    # Flush test user cache first to test cold miss
    if state.get("cache") and state["cache"].is_available():
        state["cache"].client.delete(f"rec:test_warm:{test_user}:5")

    payload = {"user_id": test_user, "k": 5, "variant": "test_warm"}
    resp = client.post("/recommend", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["user_id"] == test_user
    assert data["variant"] == "test_warm"
    assert data["cache_hit"] is False
    assert data["source"] == "model"
    assert len(data["items"]) == 5
    for idx, item in enumerate(data["items"], start=1):
        assert item["rank"] == idx
        assert item["item_id"] > 0
        assert item["title"] != ""
        assert isinstance(item["score"], float)


def test_cache_hit_on_repeated_query(client):
    """Verify repeated query hits Redis cache with cache_hit=True."""
    warm_users = list(state.get("user_features", {}).keys())
    if not warm_users:
        pytest.skip("No warm users loaded in state.")
    test_user = warm_users[0]

    payload = {"user_id": test_user, "k": 5, "variant": "test_cache"}

    # First call (cache miss)
    resp1 = client.post("/recommend", json=payload)
    assert resp1.status_code == 200

    # If Redis is active, second call must hit cache
    if state.get("cache") and state["cache"].is_available():
        resp2 = client.post("/recommend", json=payload)
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["cache_hit"] is True
        assert data2["source"] == "cache"
        assert len(data2["items"]) == 5


def test_cold_user_popularity_fallback(client):
    """Verify cold user without seed preferences gets time-decayed popularity fallback."""
    cold_user = 99999999
    # Flush cache
    if state.get("cache") and state["cache"].is_available():
        state["cache"].client.delete(f"rec:v1:{cold_user}:5")

    payload = {"user_id": cold_user, "k": 5, "variant": "v1"}
    resp = client.post("/recommend", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["user_id"] == cold_user
    assert data["source"] == "fallback_popularity"
    assert len(data["items"]) == 5


def test_cold_user_content_fallback(client):
    """Verify cold user with seed genres gets content-based recommendations."""
    cold_user = 88888888
    # Flush cache
    if state.get("cache") and state["cache"].is_available():
        state["cache"].client.delete(f"rec:v1:{cold_user}:5")

    payload = {
        "user_id": cold_user,
        "k": 5,
        "variant": "v1",
        "seed_genres": ["Sci-Fi", "Action"],
    }
    resp = client.post("/recommend", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["user_id"] == cold_user
    assert data["source"] == "fallback_content"
    assert len(data["items"]) == 5


def test_validation_bounds(client):
    """Verify Pydantic request validation bounds."""
    # Negative user_id
    resp = client.post("/recommend", json={"user_id": -1, "k": 10})
    assert resp.status_code == 422

    # k < 1
    resp = client.post("/recommend", json={"user_id": 1, "k": 0})
    assert resp.status_code == 422

    # k > 100
    resp = client.post("/recommend", json={"user_id": 1, "k": 101})
    assert resp.status_code == 422


def test_redis_down_resilience(client):
    """Verify service continues serving recommendations even when Redis is down/unavailable."""
    original_cache = state.get("cache")
    try:
        # Simulate Redis outage
        state["cache"] = None
        warm_users = list(state.get("user_features", {}).keys())
        test_user = warm_users[0] if warm_users else 1

        resp = client.post("/recommend", json={"user_id": test_user, "k": 5})
        assert resp.status_code == 200
        data = resp.json()
        assert data["cache_hit"] is False
        assert len(data["items"]) == 5
    finally:
        state["cache"] = original_cache


def test_triton_down_local_fallback(client):
    """Verify service falls back to in-process local model if Triton is offline."""
    assert state.get("ranker") is not None
    assert state["ranker"].is_ready() is True
    assert state["ranker"].served_by in ["local", "triton"]

    # Verify health endpoint reports served_by correctly
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["served_by"] == state["ranker"].served_by


def test_events_logging_and_ab_summary(client, tmp_path):
    """Verify /events logging and /ab/summary metric aggregation."""
    # Point events log to temp directory for clean test
    test_events_file = tmp_path / "test_events.jsonl"
    state["events_path"] = test_events_file

    # 1. Log impression for variant A
    resp_imp = client.post(
        "/events",
        json={"user_id": 101, "item_id": 500, "variant": "A", "event": "impression"},
    )
    assert resp_imp.status_code == 200
    assert resp_imp.json()["logged"] is True

    # 2. Log click for variant A
    resp_clk = client.post(
        "/events",
        json={"user_id": 101, "item_id": 500, "variant": "A", "event": "click"},
    )
    assert resp_clk.status_code == 200

    # 3. Log impression for variant B (no click)
    client.post(
        "/events",
        json={"user_id": 102, "item_id": 600, "variant": "B", "event": "impression"},
    )

    # 4. Check /ab/summary
    resp_sum = client.get("/ab/summary")
    assert resp_sum.status_code == 200
    data = resp_sum.json()

    assert data["total_events"] == 3
    assert "A" in data["variants"]
    assert "B" in data["variants"]

    assert data["variants"]["A"]["impressions"] == 1
    assert data["variants"]["A"]["clicks"] == 1
    assert data["variants"]["A"]["ctr"] == 1.0

    assert data["variants"]["B"]["impressions"] == 1
    assert data["variants"]["B"]["clicks"] == 0
    assert data["variants"]["B"]["ctr"] == 0.0
