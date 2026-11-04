"""
A/B Test Replay Simulation Module.
Simulates online user traffic using held-out test interactions as ground-truth user engagement.
Calculates impressions, simulated clicks, Click-Through Rate (CTR), relative lift,
and 95% bootstrap confidence intervals across Variant A (retrieval-only) and Variant B (two-stage).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Tuple

import numpy as np

from src.reco.serving.ab import get_experiment_variant

logger = logging.getLogger(__name__)


def _normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function using math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_bootstrap_confidence_interval(
    clicks_a: np.ndarray,
    clicks_b: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute 95% bootstrap confidence interval for absolute and relative difference in CTR.

    Args:
        clicks_a: 1D binary array (1 for click, 0 for no click) for Variant A
        clicks_b: 1D binary array for Variant B
        n_bootstrap: number of bootstrap resamples (default: 1000)
        alpha: significance level (0.05 for 95% CI)
        seed: random seed for reproducibility

    Returns:
        dict with mean_a, mean_b, abs_diff, rel_lift_pct, ci_lower, ci_upper, p_value
    """
    rng = np.random.default_rng(seed)
    n_a = len(clicks_a)
    n_b = len(clicks_b)

    if n_a == 0 or n_b == 0:
        return {
            "ctr_a": 0.0,
            "ctr_b": 0.0,
            "absolute_diff": 0.0,
            "relative_lift_pct": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "p_value": 1.0,
        }

    ctr_a = float(np.mean(clicks_a))
    ctr_b = float(np.mean(clicks_b))
    abs_diff = ctr_b - ctr_a
    rel_lift = ((ctr_b - ctr_a) / ctr_a * 100.0) if ctr_a > 0 else 0.0

    # Resampling
    diffs = []
    for _ in range(n_bootstrap):
        sample_a = rng.choice(clicks_a, size=n_a, replace=True)
        sample_b = rng.choice(clicks_b, size=n_b, replace=True)
        diffs.append(float(np.mean(sample_b) - np.mean(sample_a)))

    diffs_arr = np.array(diffs)
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(diffs_arr, lower_pct))
    ci_upper = float(np.percentile(diffs_arr, upper_pct))

    # Two-sample z-test for proportion difference
    pooled_p = (np.sum(clicks_a) + np.sum(clicks_b)) / (n_a + n_b)
    se_pooled = np.sqrt(pooled_p * (1.0 - pooled_p) * (1.0 / n_a + 1.0 / n_b)) if pooled_p > 0 else 1.0
    z_stat = abs_diff / np.maximum(se_pooled, 1e-12)
    p_val = float(2.0 * (1.0 - _normal_cdf(abs(float(z_stat)))))

    return {
        "ctr_a": round(ctr_a, 5),
        "ctr_b": round(ctr_b, 5),
        "absolute_diff": round(abs_diff, 5),
        "relative_lift_pct": round(rel_lift, 2),
        "ci_lower_abs": round(ci_lower, 5),
        "ci_upper_abs": round(ci_upper, 5),
        "ci_lower_rel_pct": round((ci_lower / ctr_a * 100.0) if ctr_a > 0 else 0.0, 2),
        "ci_upper_rel_pct": round((ci_upper / ctr_a * 100.0) if ctr_a > 0 else 0.0, 2),
        "z_statistic": round(float(z_stat), 4),
        "p_value": round(p_val, 6),
        "statistically_significant": bool(p_val < alpha),
    }


def simulate_ab_replay(
    sessions: List[Tuple[int, int, List[int], List[int]]],
    k: int = 10,
    salt: str = "recoengine_ab_2026",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run A/B replay simulation over evaluation sessions.

    Args:
        sessions: list of tuples: (user_id, true_positive_item_id, variant_a_topk_items, variant_b_topk_items)
        k: recommendation list length (default: 10)
        salt: experiment salt for deterministic bucketing
        n_bootstrap: number of bootstrap iterations
        seed: random seed

    Returns:
        Structured simulation results dictionary with impressions, clicks, CTR, and bootstrap CIs.
    """
    variant_a_clicks = []
    variant_b_clicks = []

    for user_id, true_item, topk_a, topk_b in sessions:
        variant = get_experiment_variant(user_id=user_id, salt=salt)
        if variant == "A":
            hit = 1 if true_item in topk_a[:k] else 0
            variant_a_clicks.append(hit)
        else:
            hit = 1 if true_item in topk_b[:k] else 0
            variant_b_clicks.append(hit)

    arr_a = np.array(variant_a_clicks, dtype=np.int32)
    arr_b = np.array(variant_b_clicks, dtype=np.int32)

    ci_stats = compute_bootstrap_confidence_interval(
        clicks_a=arr_a,
        clicks_b=arr_b,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    results = {
        "simulation_method": "offline_replay_simulation",
        "description": "Replay simulation using held-out positive test interactions as simulated clicks",
        "sample_size": len(sessions),
        "k": k,
        "variant_a": {
            "name": "Variant A (Retrieval-Only)",
            "impressions": int(len(arr_a)),
            "clicks": int(np.sum(arr_a)),
            "ctr": ci_stats["ctr_a"],
        },
        "variant_b": {
            "name": "Variant B (Retrieval + DCN-V2 Ranker)",
            "impressions": int(len(arr_b)),
            "clicks": int(np.sum(arr_b)),
            "ctr": ci_stats["ctr_b"],
        },
        "comparison": {
            "absolute_diff": ci_stats["absolute_diff"],
            "relative_lift_pct": ci_stats["relative_lift_pct"],
            "95_ci_abs": [ci_stats["ci_lower_abs"], ci_stats["ci_upper_abs"]],
            "95_ci_rel_pct": [ci_stats["ci_lower_rel_pct"], ci_stats["ci_upper_rel_pct"]],
            "z_statistic": ci_stats["z_statistic"],
            "p_value": ci_stats["p_value"],
            "statistically_significant": ci_stats["statistically_significant"],
        },
    }
    logger.info("A/B Replay Simulation Completed: %d sessions | Variant A CTR: %.4f | Variant B CTR: %.4f | Lift: +%.2f%% (p=%.6f)",
                len(sessions), ci_stats["ctr_a"], ci_stats["ctr_b"], ci_stats["relative_lift_pct"], ci_stats["p_value"])
    return results
