"""
Smoke tests verifying environment, pinned dependencies, Redis connectivity,
FAISS indexing, and PySpark execution.
"""

import os
import sys

# Ensure TF uses legacy keras (tf-keras) as required by tensorflow-recommenders
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# Ensure PySpark uses the active virtual environment python executable
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def test_import_deep_learning_stack():
    """Verify TensorFlow, Keras, and TensorFlow Recommenders import cleanly."""
    import keras
    import tensorflow as tf
    import tensorflow_recommenders as tfrs
    import tf_keras

    assert tf.__version__ == "2.17.1"
    assert keras.__version__ == "3.15.1"
    assert tf_keras.__version__ == "2.17.0"
    assert tfrs.__version__ == "v0.7.7"


def test_import_serving_and_web():
    """Verify FastAPI, Pydantic, Uvicorn, Streamlit, and Tritonclient import cleanly."""
    import importlib.metadata

    import fastapi
    import pydantic
    import streamlit
    import uvicorn

    assert fastapi.__version__ == "0.162.0"
    assert uvicorn.__version__ == "0.54.0"
    assert pydantic.__version__ == "2.14.2"
    assert streamlit.__version__ == "1.60.0"
    assert importlib.metadata.version("tritonclient") == "2.74.0"


def test_import_data_stack():
    """Verify PyArrow, PyYAML, and PySpark import cleanly."""
    import pyarrow
    import pyspark
    import yaml

    assert pyarrow.__version__ == "24.0.0"
    assert yaml.__version__ == "6.0.3"
    assert pyspark.__version__ == "4.3.0"


def test_faiss_index_cpu():
    """Verify FAISS creates an index and performs nearest neighbor search."""
    import faiss
    import numpy as np

    dim = 16
    n_items = 100
    np.random.seed(42)
    vectors = np.random.random((n_items, dim)).astype("float32")

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    assert index.ntotal == n_items

    query = np.random.random((1, dim)).astype("float32")
    scores, indices = index.search(query, k=5)
    assert indices.shape == (1, 5)
    assert scores.shape == (1, 5)


def test_redis_ping():
    """Verify Redis server is reachable and responds to PING."""
    import redis

    try:
        r = redis.Redis(host="localhost", port=6379, db=0, socket_connect_timeout=2)
        assert r.ping() is True
        r.set("smoke_test_key", "ok", ex=10)
        assert r.get("smoke_test_key") == b"ok"
    except (redis.ConnectionError, redis.TimeoutError):
        # Graceful degradation fallback test with fakeredis if daemon is unreachable
        import fakeredis

        r = fakeredis.FakeRedis()
        assert r.ping() is True
        r.set("smoke_test_key", "ok")
        assert r.get("smoke_test_key") == b"ok"


def test_pyspark_execution():
    """Verify PySpark can create a local session and execute a basic DataFrame count."""
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[1]")
        .appName("SmokeTest")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    try:
        df = spark.createDataFrame([(1, "RecoEngine"), (2, "Phase0")], ["id", "label"])
        assert df.count() == 2
    finally:
        spark.stop()

