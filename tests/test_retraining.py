"""
Unit and Integration Tests for Retraining Pipeline and Model Promotion Gate.
Tests:
1. Airflow DAG import and dependency graph structure
2. Promotion gate decision logic (pass, tolerance pass, rejection)
3. End-to-end retraining pipeline execution and gate rejection simulation
4. Redis cache invalidation on model promotion
"""

from __future__ import annotations

from pathlib import Path

import pytest

from airflow.dags.retrain_dag import dag
from scripts.run_retraining import run_pipeline
from src.reco.retraining.gate import evaluate_promotion_gate
from src.reco.serving.cache import RedisCache

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_retrain_dag_structure():
    """Verify DAG structure, task presence, and sequential dependency graph."""
    assert dag is not None
    assert dag.dag_id == "recoengine_retrain_pipeline"

    expected_tasks = [
        "build_features",
        "train_retrieval",
        "build_index",
        "train_ranker",
        "evaluate_candidate",
        "promotion_gate",
        "export_artifacts",
        "flush_cache",
    ]
    for task_id in expected_tasks:
        assert task_id in dag.tasks, f"Task {task_id} missing from DAG."

    # Verify task ordering
    build_feat = dag.tasks["build_features"]
    train_ret = dag.tasks["train_retrieval"]
    build_idx = dag.tasks["build_index"]
    train_rnk = dag.tasks["train_ranker"]
    eval_cand = dag.tasks["evaluate_candidate"]
    gate = dag.tasks["promotion_gate"]
    export_art = dag.tasks["export_artifacts"]
    flush = dag.tasks["flush_cache"]

    assert train_ret in build_feat.downstream
    assert build_idx in train_ret.downstream
    assert train_rnk in build_idx.downstream
    assert eval_cand in train_rnk.downstream
    assert gate in eval_cand.downstream
    assert export_art in gate.downstream
    assert flush in export_art.downstream


def test_promotion_gate_logic():
    """Verify promotion gate evaluation under superior, non-inferior, and inferior candidates."""
    prod = {"ndcg@10": 0.03500}

    # Case 1: Strictly superior candidate
    cand_better = {"ndcg@10": 0.03800}
    dec1 = evaluate_promotion_gate(cand_better, prod, tolerance=0.001)
    assert dec1["promoted"] is True
    assert dec1["delta"] > 0

    # Case 2: Candidate within tolerance margin (e.g. -0.0005 with 0.001 tolerance)
    cand_tol = {"ndcg@10": 0.03450}
    dec2 = evaluate_promotion_gate(cand_tol, prod, tolerance=0.001)
    assert dec2["promoted"] is True

    # Case 3: Degraded candidate failing gate
    cand_degraded = {"ndcg@10": 0.03200}
    dec3 = evaluate_promotion_gate(cand_degraded, prod, tolerance=0.001)
    assert dec3["promoted"] is False
    assert "REJECTED" in dec3["reason"]


def test_retraining_pipeline_successful_run(tmp_path):
    """Verify end-to-end retraining execution with successful gate promotion."""
    result = run_pipeline(
        use_dev=True,
        simulate_rejection=False,
        tolerance=0.001,
        target_version="v2_test",
    )
    assert result["gate_decision"]["promoted"] is True
    assert result["action_taken"]["status"] == "promoted"
    assert result["action_taken"]["promoted_version"] == "v2_test"

    # Verify target artifacts directory was populated
    target_dir = PROJECT_ROOT / "artifacts" / "v2_test"
    assert target_dir.exists()
    assert (target_dir / "faiss.index").exists()

    # Clean up test artifact directory
    import shutil

    shutil.rmtree(target_dir, ignore_errors=True)


def test_retraining_pipeline_rejection_run():
    """Verify end-to-end retraining execution with simulated gate rejection."""
    result = run_pipeline(
        use_dev=True,
        simulate_rejection=True,
        tolerance=0.001,
        target_version="v_rejected",
    )
    assert result["gate_decision"]["promoted"] is False
    assert result["action_taken"]["status"] == "rejected"
    assert result["action_taken"]["active_version"] == "v1"

    # Verify rejected target artifact was NOT created
    target_dir = PROJECT_ROOT / "artifacts" / "v_rejected"
    assert not target_dir.exists()


def test_cache_flush_on_promotion():
    """Verify Redis cache invalidates variant keys on promotion."""
    cache = RedisCache()
    if not cache.is_available():
        pytest.skip("Live Redis not available for cache test.")

    # Populate dummy keys for v1
    cache.set_recommendations("v1", 12345, 10, [{"item_id": 1, "score": 0.9}])
    assert cache.get_recommendations("v1", 12345, 10) is not None

    # Invalidate
    flushed = cache.flush_variant("v1")
    assert flushed >= 1
    assert cache.get_recommendations("v1", 12345, 10) is None
