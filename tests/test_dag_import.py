"""
Airflow DAG import test.
Verifies that the retraining DAG imports without syntax or configuration errors.
"""

from __future__ import annotations


def test_dag_imports_without_errors():
    """Verify retrain_dag module imports cleanly and exposes valid dag object."""
    from airflow.dags.retrain_dag import dag

    assert dag is not None
    assert dag.dag_id == "recoengine_retrain_pipeline"
    assert len(dag.tasks) == 8

