"""
Unit tests for evaluation metrics against hand-computed ground truth.
"""

import math

import pytest

from src.reco.evaluation.metrics import (
    evaluate_recommendations,
    ndcg_at_k,
    recall_at_k,
)


def test_hand_computed_metrics():
    """Verify Recall and NDCG against exact mathematical calculations."""
    gt = {10, 20}
    rec = [10, 30, 20, 40, 50]

    # K = 1
    assert recall_at_k(rec, gt, k=1) == pytest.approx(0.5, abs=1e-6)
    assert ndcg_at_k(rec, gt, k=1) == pytest.approx(1.0, abs=1e-6)

    # K = 2
    idcg_2 = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    dcg_2 = 1.0 / math.log2(2)
    expected_ndcg_2 = dcg_2 / idcg_2
    assert recall_at_k(rec, gt, k=2) == pytest.approx(0.5, abs=1e-6)
    assert ndcg_at_k(rec, gt, k=2) == pytest.approx(expected_ndcg_2, abs=1e-6)

    # K = 3
    dcg_3 = 1.0 / math.log2(2) + 1.0 / math.log2(4)  # 1.0 + 0.5 = 1.5
    idcg_3 = idcg_2  # len(gt) = 2, min(3, 2) = 2
    expected_ndcg_3 = dcg_3 / idcg_3
    assert recall_at_k(rec, gt, k=3) == pytest.approx(1.0, abs=1e-6)
    assert ndcg_at_k(rec, gt, k=3) == pytest.approx(expected_ndcg_3, abs=1e-6)


def test_empty_cases():
    """Verify metrics return 0.0 on empty inputs without throwing exceptions."""
    assert recall_at_k([], {1, 2}, k=10) == 0.0
    assert recall_at_k([1, 2], set(), k=10) == 0.0
    assert ndcg_at_k([], {1, 2}, k=10) == 0.0
    assert ndcg_at_k([1, 2], set(), k=10) == 0.0
    assert recall_at_k([1, 2], {1, 2}, k=0) == 0.0
    assert ndcg_at_k([1, 2], {1, 2}, k=0) == 0.0


def test_macro_evaluation():
    """Verify multi-user aggregated evaluation calculation."""
    recs = {
        1: [10, 20, 30],
        2: [40, 50, 60],
    }
    gt = {
        1: {10, 20},  # recall@2 = 1.0
        2: {99},      # recall@2 = 0.0
    }
    results = evaluate_recommendations(recs, gt, k_list=(2,))
    assert "recall@2" in results
    assert "ndcg@2" in results
    assert results["recall@2"] == pytest.approx(0.5, abs=1e-6)
