"""
Model Promotion Gate for Retraining Pipeline.
Evaluates candidate model performance against production baseline on held-out validation set.
Only promotes if candidate NDCG@10 >= production NDCG@10 - tolerance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def evaluate_promotion_gate(
    candidate_metrics: Dict[str, Any],
    production_metrics: Dict[str, Any],
    metric_key: str = "ndcg@10",
    tolerance: float = 0.0,
) -> Dict[str, Any]:
    """
    Decide whether candidate model should be promoted over production model.

    Args:
        candidate_metrics: dict containing evaluation metrics for new candidate model
        production_metrics: dict containing evaluation metrics for current production model
        metric_key: metric name to evaluate gate on (default: 'ndcg@10')
        tolerance: allowed degradation margin (default: 0.0, strict non-inferiority)

    Returns:
        decision dict with:
        - 'promoted': bool
        - 'candidate_score': float
        - 'production_score': float
        - 'delta': float
        - 'reason': str
    """
    candidate_score = float(candidate_metrics.get(metric_key, 0.0))
    production_score = float(production_metrics.get(metric_key, 0.0))
    delta = candidate_score - production_score
    threshold = production_score - tolerance

    promoted = candidate_score >= threshold

    if promoted:
        reason = (
            f"Candidate model promoted: {metric_key} ({candidate_score:.5f}) >= "
            f"threshold ({threshold:.5f}) [production: {production_score:.5f}, delta: {delta:+.5f}]."
        )
        logger.info(reason)
    else:
        reason = (
            f"Candidate model REJECTED: {metric_key} ({candidate_score:.5f}) < "
            f"threshold ({threshold:.5f}) [production: {production_score:.5f}, delta: {delta:+.5f}]."
        )
        logger.warning(reason)

    return {
        "promoted": promoted,
        "metric_key": metric_key,
        "candidate_score": round(candidate_score, 5),
        "production_score": round(production_score, 5),
        "delta": round(delta, 5),
        "tolerance": tolerance,
        "reason": reason,
    }

