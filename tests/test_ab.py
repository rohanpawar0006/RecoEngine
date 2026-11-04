"""
Unit Tests for A/B Testing Deterministic Bucketing and Replay Simulation.
"""

from __future__ import annotations

import numpy as np

from src.reco.evaluation.ab_replay import (
    compute_bootstrap_confidence_interval,
    simulate_ab_replay,
)
from src.reco.serving.ab import (
    get_experiment_variant,
    get_user_hash_ratio,
    get_variant_description,
)


def test_user_hash_ratio_bounds_and_determinism():
    """Verify hash ratio is deterministic and strictly within [0.0, 1.0)."""
    for uid in [1, 42, 100, 999999]:
        r1 = get_user_hash_ratio(uid)
        r2 = get_user_hash_ratio(uid)
        assert r1 == r2
        assert 0.0 <= r1 < 1.0


def test_experiment_variant_determinism():
    """Verify same user ID always maps to the same variant."""
    for uid in range(1, 50):
        v1 = get_experiment_variant(uid)
        v2 = get_experiment_variant(uid)
        assert v1 == v2
        assert v1 in ["A", "B"]


def test_experiment_variant_population_balance():
    """Verify traffic is evenly balanced ~50/50 across a population of users."""
    n_users = 10000
    variants = [get_experiment_variant(uid) for uid in range(1, n_users + 1)]
    count_a = variants.count("A")
    ratio_a = count_a / n_users

    # Should be very close to 0.50 (within 0.48 - 0.52)
    assert 0.48 <= ratio_a <= 0.52, f"Ratio {ratio_a} out of expected 50/50 balance bounds."


def test_variant_description():
    """Verify metadata descriptions for variants A and B."""
    desc_a = get_variant_description("A")
    assert desc_a["variant"] == "A"
    assert "Retrieval-Only" in desc_a["name"]

    desc_b = get_variant_description("B")
    assert desc_b["variant"] == "B"
    assert "DCN-V2" in desc_b["name"]


def test_bootstrap_confidence_interval():
    """Verify bootstrap CI computation on synthetic click arrays."""
    # Synthetic: Variant B is distinctly better than Variant A
    clicks_a = np.array([1] * 50 + [0] * 950)   # CTR = 0.05
    clicks_b = np.array([1] * 100 + [0] * 900)  # CTR = 0.10

    res = compute_bootstrap_confidence_interval(
        clicks_a=clicks_a,
        clicks_b=clicks_b,
        n_bootstrap=500,
        seed=42,
    )

    assert res["ctr_a"] == 0.05
    assert res["ctr_b"] == 0.10
    assert res["relative_lift_pct"] == 100.0
    assert res["absolute_diff"] == 0.05
    assert res["ci_lower_abs"] > 0.0  # Significantly positive lift
    assert res["p_value"] < 0.05
    assert res["statistically_significant"] is True


def test_simulate_ab_replay_accounting():
    """Verify simulation session accounting and hit calculation."""
    # 4 sessions:
    # User 1: true_item 101, topk_a [101, 102], topk_b [103, 104]
    # User 2: true_item 201, topk_a [202, 203], topk_b [201, 204]
    sessions = [
        (1, 101, [101, 102], [103, 104]),
        (2, 201, [202, 203], [201, 204]),
        (3, 301, [301, 302], [301, 303]),
        (4, 401, [402, 403], [404, 405]),
    ]

    sim = simulate_ab_replay(sessions=sessions, k=2, n_bootstrap=200)
    assert sim["sample_size"] == 4
    total_impressions = sim["variant_a"]["impressions"] + sim["variant_b"]["impressions"]
    assert total_impressions == 4
    total_clicks = sim["variant_a"]["clicks"] + sim["variant_b"]["clicks"]
    assert 0 <= total_clicks <= 4

