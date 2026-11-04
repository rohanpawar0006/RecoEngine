"""
Two-Tower Retrieval Model using TensorFlow Recommenders.
Consists of:
- UserTower: maps user index to a normalized D-dimensional embedding vector.
- ItemTower: maps item index to a normalized D-dimensional embedding vector.
- TwoTowerModel: end-to-end model trained with in-batch negative sampled softmax loss.
"""

import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"


import tensorflow as tf
import tensorflow_recommenders as tfrs
import tf_keras as keras


class UserTower(keras.Model):
    """User tower neural network producing normalized user embeddings."""

    def __init__(
        self,
        num_users: int,
        embedding_dim: int = 64,
        dense_units: list[int] | None = None,
        name: str = "user_tower",
    ) -> None:
        super().__init__(name=name)
        self.num_users = num_users
        self.embedding_dim = embedding_dim
        self.embedding = keras.layers.Embedding(
            input_dim=num_users,
            output_dim=embedding_dim,
            embeddings_initializer="glorot_uniform",
            name="user_embedding",
        )
        self.dense_layers = []
        if dense_units:
            for i, units in enumerate(dense_units):
                self.dense_layers.append(
                    keras.layers.Dense(units, activation="relu", name=f"user_dense_{i}")
                )
        self.projection = keras.layers.Dense(embedding_dim, activation=None, name="user_projection")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass for user inputs (tensor of user indices)."""
        x = self.embedding(inputs)
        for layer in self.dense_layers:
            x = layer(x)
        x = self.projection(x)
        return tf.linalg.l2_normalize(x, axis=-1)

    def get_config(self):
        return {
            "num_users": self.num_users,
            "embedding_dim": self.embedding_dim,
        }


class ItemTower(keras.Model):
    """Item tower neural network producing normalized item embeddings."""

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        dense_units: list[int] | None = None,
        name: str = "item_tower",
    ) -> None:
        super().__init__(name=name)
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.embedding = keras.layers.Embedding(
            input_dim=num_items,
            output_dim=embedding_dim,
            embeddings_initializer="glorot_uniform",
            name="item_embedding",
        )
        self.dense_layers = []
        if dense_units:
            for i, units in enumerate(dense_units):
                self.dense_layers.append(
                    keras.layers.Dense(units, activation="relu", name=f"item_dense_{i}")
                )
        self.projection = keras.layers.Dense(embedding_dim, activation=None, name="item_projection")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass for item inputs (tensor of item indices)."""
        x = self.embedding(inputs)
        for layer in self.dense_layers:
            x = layer(x)
        x = self.projection(x)
        return tf.linalg.l2_normalize(x, axis=-1)

    def get_config(self):
        return {
            "num_items": self.num_items,
            "embedding_dim": self.embedding_dim,
        }


class TwoTowerModel(tfrs.Model):
    """
    Two-Tower Recommendation Model.
    Optimizes candidate retrieval using in-batch negative sampled softmax loss.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        temperature: float = 0.05,
    ) -> None:
        super().__init__()
        self.user_tower = UserTower(num_users=num_users, embedding_dim=embedding_dim)
        self.item_tower = ItemTower(num_items=num_items, embedding_dim=embedding_dim)
        self.task = tfrs.tasks.Retrieval(temperature=temperature)

    def compute_loss(self, features: dict, training: bool = False) -> tf.Tensor:
        """Compute the in-batch sampled softmax retrieval loss."""
        user_embeddings = self.user_tower(features["userId"])
        item_embeddings = self.item_tower(features["movieId"])
        return self.task(user_embeddings, item_embeddings)
