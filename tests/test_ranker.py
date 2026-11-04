"""
Unit tests for Deep Ranking Model (DCN-V2).
Tests input/output shapes, score probability bounds, and SavedModel reloadability.
"""

import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tempfile

import numpy as np
import pytest
import tensorflow as tf
import tf_keras as keras

from src.reco.ranking.model import DCNv2Ranker
from src.reco.utils.seeds import set_seed


@pytest.fixture
def dummy_ranker():
    """Create a lightweight DCN-V2 ranker instance."""
    set_seed(42)
    model = DCNv2Ranker(
        num_users=50,
        num_items=100,
        embedding_dim=16,
        num_cross_layers=2,
        deep_units=[32, 16],
        dropout_rate=0.0,
    )
    return model


def test_ranker_input_output_shapes(dummy_ranker):
    """Verify ranker input dictionary parsing and output probability shape and bounds."""
    batch_size = 8
    inputs = {
        "userId": tf.constant([1, 2, 3, 4, 5, 6, 7, 8], dtype=tf.int32),
        "movieId": tf.constant([10, 20, 30, 40, 50, 60, 70, 80], dtype=tf.int32),
        "user_features": tf.random.uniform((batch_size, 2), dtype=tf.float32),
        "item_features": tf.random.uniform((batch_size, 3), dtype=tf.float32),
    }

    scores = dummy_ranker(inputs).numpy()

    assert scores.shape == (batch_size,)
    assert not np.isnan(scores).any()
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


def test_score_monotonic_sanity(dummy_ranker):
    """Verify that predictions are consistent and finite across varied item quality features."""
    user = tf.constant([1, 1], dtype=tf.int32)
    movie = tf.constant([10, 10], dtype=tf.int32)
    u_feat = tf.constant([[5.0, 4.0], [5.0, 4.0]], dtype=tf.float32)
    # Compare low vs high item quality features
    i_feat = tf.constant([[1.0, 1.0, 0.5], [10.0, 5.0, 0.9]], dtype=tf.float32)

    inputs = {
        "userId": user,
        "movieId": movie,
        "user_features": u_feat,
        "item_features": i_feat,
    }

    scores = dummy_ranker(inputs).numpy()
    assert len(scores) == 2
    assert scores[0] != scores[1]  # Features clearly impact predicted scores


def test_ranker_save_load_roundtrip(dummy_ranker):
    """Verify exported SavedModel can be reloaded and yields identical prediction outputs."""
    inputs = {
        "userId": tf.constant([5, 12], dtype=tf.int32),
        "movieId": tf.constant([25, 75], dtype=tf.int32),
        "user_features": tf.constant([[2.0, 3.5], [4.0, 4.2]], dtype=tf.float32),
        "item_features": tf.constant([[5.0, 3.8, 0.8], [8.0, 4.5, 0.95]], dtype=tf.float32),
    }

    original_scores = dummy_ranker(inputs).numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = os.path.join(tmpdir, "test_ranker")
        dummy_ranker.save(export_path, save_format="tf")

        loaded_ranker = keras.models.load_model(export_path)
        loaded_scores = loaded_ranker(inputs).numpy()

        np.testing.assert_allclose(original_scores, loaded_scores, atol=1e-5)
