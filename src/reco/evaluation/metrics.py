"""
Evaluation metrics for recommendation systems.
Computes Recall@K and NDCG@K with binary or graded relevance.
Tested against hand-computed ground truth in tests/test_metrics.py.
"""

from collections.abc import Sequence

import numpy as np


def recall_at_k(recommended: Sequence[int], ground_truth: set[int], k: int) -> float:
    """
    Compute Recall@K for a single user.

    Args:
        recommended: Ordered sequence of recommended item IDs (top ranked first).
        ground_truth: Set of ground-truth positive item IDs.
        k: Cut-off rank.

    Returns:
        Recall@K score in [0.0, 1.0]. Returns 0.0 if ground_truth is empty.
    """
    if not ground_truth or k <= 0:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in ground_truth)
    return hits / float(len(ground_truth))


def ndcg_at_k(recommended: Sequence[int], ground_truth: set[int], k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@K) for a single user with binary relevance.

    Args:
        recommended: Ordered sequence of recommended item IDs (top ranked first).
        ground_truth: Set of ground-truth positive item IDs.
        k: Cut-off rank.

    Returns:
        NDCG@K score in [0.0, 1.0]. Returns 0.0 if ground_truth is empty.
    """
    if not ground_truth or k <= 0:
        return 0.0

    rec_k = recommended[:k]
    dcg = 0.0
    for idx, item in enumerate(rec_k):
        if item in ground_truth:
            # Rank position is 1-indexed: idx + 1
            # Discount is log2(rank + 1) = log2((idx + 1) + 1) = log2(idx + 2)
            dcg += 1.0 / np.log2(idx + 2)

    # Ideal DCG: top min(k, len(ground_truth)) items all relevant
    ideal_hits = min(k, len(ground_truth))
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return float(dcg / idcg)


def evaluate_recommendations(
    recommendations: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    k_list: Sequence[int] = (10, 50, 200),
) -> dict[str, float]:
    """
    Compute macro-averaged Recall@K and NDCG@K across all evaluated users.

    Args:
        recommendations: Mapping from user_id to ordered list of recommended item_ids.
        ground_truth: Mapping from user_id to set of positive item_ids.
        k_list: Tuple or list of cutoff ranks.

    Returns:
        Dictionary of mean metric scores (e.g. {'recall@10': 0.05, 'ndcg@10': 0.08, ...}).
    """
    evaluated_users = [u for u in ground_truth if len(ground_truth[u]) > 0 and u in recommendations]
    if not evaluated_users:
        return {f"{m}@{k}": 0.0 for k in k_list for m in ("recall", "ndcg")}

    results: dict[str, float] = {}
    for k in k_list:
        recalls = [
            recall_at_k(recommendations[u], ground_truth[u], k) for u in evaluated_users
        ]
        ndcgs = [
            ndcg_at_k(recommendations[u], ground_truth[u], k) for u in evaluated_users
        ]
        results[f"recall@{k}"] = float(np.mean(recalls))
        results[f"ndcg@{k}"] = float(np.mean(ndcgs))

    return results
