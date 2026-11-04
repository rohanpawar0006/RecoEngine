"""
Unit tests for Two-Tower Retrieval Model.
Tests embedding shapes, normalization, finiteness, determinism, and loss computation.
"""

import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import tensorflow as tf

from src.reco.retrieval.model import ItemTower, TwoTowerModel, UserTower
from src.reco.utils.seeds import set_seed


def test_user_and_item_tower_shapes():
    """Verify output embedding shapes match the configured embedding dimension."""
    num_users = 50
    num_items = 100
    dim = 32

    user_tower = UserTower(num_users=num_users, embedding_dim=dim)
    item_tower = ItemTower(num_items=num_items, embedding_dim=dim)

    batch_users = tf.constant([0, 5, 12, 49], dtype=tf.int32)
    batch_items = tf.constant([1, 10, 55, 99], dtype=tf.int32)

    u_emb = user_tower(batch_users)
    i_emb = item_tower(batch_items)

    assert u_emb.shape == (4, dim)
    assert i_emb.shape == (4, dim)


def test_embeddings_normalized_and_finite():
    """Verify that all produced embeddings are finite and L2-normalized to unit length."""
    user_tower = UserTower(num_users=20, embedding_dim=16)
    inputs = tf.constant([1, 2, 3], dtype=tf.int32)
    embeddings = user_tower(inputs).numpy()

    assert not np.isnan(embeddings).any(), "Found NaN in embeddings"
    assert not np.isinf(embeddings).any(), "Found Inf in embeddings"

    # Verify unit length (L2 norm == 1.0)
    norms = np.linalg.norm(embeddings, axis=-1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)


def test_deterministic_output_for_fixed_seed():
    """Verify that two models initialized with identical seeds generate identical embeddings."""
    set_seed(42)
    tower1 = UserTower(num_users=20, embedding_dim=16)
    _ = tower1(tf.constant([1]))

    set_seed(42)
    tower2 = UserTower(num_users=20, embedding_dim=16)
    _ = tower2(tf.constant([1]))

    inputs = tf.constant([2, 5, 8], dtype=tf.int32)
    emb1 = tower1(inputs).numpy()
    emb2 = tower2(inputs).numpy()

    np.testing.assert_allclose(emb1, emb2, atol=1e-6)


def test_two_tower_loss_computation():
    """Verify in-batch negative retrieval loss computation."""
    set_seed(42)
    model = TwoTowerModel(num_users=30, num_items=30, embedding_dim=16, temperature=0.05)
    batch = {
        "userId": tf.constant([1, 2, 3, 4], dtype=tf.int32),
        "movieId": tf.constant([10, 11, 12, 13], dtype=tf.int32),
    }

    loss = model.compute_loss(batch)
    assert loss is not None
    assert tf.math.is_finite(loss)
    assert loss.numpy() > 0.0
