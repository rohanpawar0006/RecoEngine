"""
Deep Ranking Model Architecture (DCN-V2 / Deep & Cross Network V2).
Combines explicit feature interaction cross-layers with deep feedforward layers
for precise CTR / positive-interaction probability prediction.
"""

import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"


import tensorflow as tf
import tf_keras as keras


class CrossLayer(keras.layers.Layer):
    """
    Explicit feature interaction layer from Deep & Cross Network (DCN).
    Formula: x_{l+1} = x_0 * (x_l @ W + b) + x_l
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.kernel = None
        self.bias = None

    def build(self, input_shape: tf.TensorShape) -> None:
        dim = input_shape[-1]
        self.kernel = self.add_weight(
            name="kernel",
            shape=(dim, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(dim,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x0: tf.Tensor, xl: tf.Tensor) -> tf.Tensor:
        """Forward pass for cross layer."""
        # x_proj shape: (B, 1)
        x_proj = tf.matmul(xl, self.kernel)
        return x0 * (x_proj + self.bias) + xl


class DCNv2Ranker(keras.Model):
    """
    DCN-V2 Ranking Model for Candidate Re-ranking.
    Predicts probability of positive user-item interaction using cross & deep networks.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 32,
        num_cross_layers: int = 2,
        deep_units: list[int] | None = None,
        dropout_rate: float = 0.2,
        name: str = "dcn_v2_ranker",
    ) -> None:
        super().__init__(name=name)
        if deep_units is None:
            deep_units = [128, 64, 32]

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        # 1. Embeddings for categorical IDs
        self.user_embedding = keras.layers.Embedding(
            input_dim=num_users,
            output_dim=embedding_dim,
            name="user_embedding",
        )
        self.item_embedding = keras.layers.Embedding(
            input_dim=num_items,
            output_dim=embedding_dim,
            name="item_embedding",
        )

        # 2. Cross Network
        self.cross_layers = [CrossLayer(name=f"cross_layer_{i}") for i in range(num_cross_layers)]

        # 3. Deep Network
        self.deep_layers = []
        for i, units in enumerate(deep_units):
            self.deep_layers.append(keras.layers.Dense(units, activation="relu", name=f"deep_{i}"))
            self.deep_layers.append(keras.layers.Dropout(dropout_rate, name=f"dropout_{i}"))

        # 4. Final Output Prediction Head
        self.prediction_head = keras.layers.Dense(1, activation="sigmoid", name="ctr_prediction")

    def call(self, inputs: dict) -> tf.Tensor:
        """
        Forward pass for ranker.
        inputs dict contains:
        - "userId": integer tensor (B,)
        - "movieId": integer tensor (B,)
        - "user_features": float tensor (B, 2) [interaction_count_log, avg_rating]
        - "item_features": float tensor (B, 3) [popularity_log, avg_rating, release_year]
        """
        u_emb = self.user_embedding(inputs["userId"])
        i_emb = self.item_embedding(inputs["movieId"])
        u_feat = inputs["user_features"]
        i_feat = inputs["item_features"]

        # Concatenate into input representation x0
        x0 = tf.concat([u_emb, i_emb, u_feat, i_feat], axis=-1)

        # Cross branch
        xl = x0
        for layer in self.cross_layers:
            xl = layer(x0, xl)

        # Deep branch
        xd = x0
        for layer in self.deep_layers:
            xd = layer(xd)

        # Combine both representations
        combined = tf.concat([xl, xd], axis=-1)
        score = self.prediction_head(combined)
        return tf.squeeze(score, axis=-1)
