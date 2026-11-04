"""
Ranker inference abstraction supporting both Triton Inference Server and in-process SavedModel fallback.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

# Suppress TF oneDNN logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

logger = logging.getLogger(__name__)


class RankerInference:
    """
    Inference client for DCN-V2 ranker.
    Tries Triton Inference Server first; seamlessly falls back to in-process TensorFlow SavedModel
    if Triton is unavailable, conforming to Scope-Cut Ladder Step 4.
    """

    def __init__(
        self,
        local_model_path: str | Path = "artifacts/v1/ranker",
        triton_url: str = "localhost:8000",
        model_name: str = "ranker",
    ):
        self.local_model_path = Path(local_model_path)
        self.triton_url = triton_url
        self.model_name = model_name
        self.served_by: str = "local"
        self.triton_client = None
        self.local_model = None

        # Attempt to connect to Triton
        self._init_triton()

        # If Triton unavailable, initialize local SavedModel
        if self.triton_client is None:
            self._init_local()

    def _init_triton(self) -> None:
        """Attempt Triton HTTP client initialization."""
        try:
            import tritonclient.http as httpclient

            client = httpclient.InferenceServerClient(url=self.triton_url, connection_timeout=0.5, network_timeout=0.5)
            if client.is_server_live() and client.is_model_ready(self.model_name):
                self.triton_client = client
                self.served_by = "triton"
                logger.info("Connected to Triton Inference Server at %s (model: %s)", self.triton_url, self.model_name)
                return
        except Exception as e:
            logger.info("Triton not active (%s). Using in-process local serving. (Details: %s)", self.triton_url, e)
        self.triton_client = None

    def _init_local(self) -> None:
        """Load in-process TensorFlow SavedModel."""
        if not self.local_model_path.exists():
            logger.warning("Local ranker SavedModel not found at %s", self.local_model_path)
            return

        try:
            import tensorflow as tf

            self.local_model = tf.saved_model.load(str(self.local_model_path))
            self.served_by = "local"
            logger.info("Loaded in-process SavedModel from %s", self.local_model_path)
        except Exception as e:
            logger.error("Failed to load local SavedModel: %s", e)
            self.local_model = None

    def is_triton_available(self) -> bool:
        return self.triton_client is not None

    def is_ready(self) -> bool:
        return (self.triton_client is not None) or (self.local_model is not None)

    def score(
        self,
        user_ids: np.ndarray,
        movie_ids: np.ndarray,
        user_features: np.ndarray,
        item_features: np.ndarray,
    ) -> np.ndarray:
        """
        Score a batch of (user, item) candidates.
        Args:
            user_ids: (N,) int32
            movie_ids: (N,) int32
            user_features: (N, 2) float32
            item_features: (N, 3) float32
        Returns:
            scores: (N,) float32 probabilities
        """
        n_items = len(user_ids)
        if n_items == 0:
            return np.empty((0,), dtype=np.float32)

        # 1. Try Triton if active
        if self.triton_client is not None:
            try:
                import tritonclient.http as httpclient

                inputs = [
                    httpclient.InferInput("userId", user_ids.shape, "INT32"),
                    httpclient.InferInput("movieId", movie_ids.shape, "INT32"),
                    httpclient.InferInput("user_features", user_features.shape, "FP32"),
                    httpclient.InferInput("item_features", item_features.shape, "FP32"),
                ]
                inputs[0].set_data_from_numpy(user_ids.astype(np.int32))
                inputs[1].set_data_from_numpy(movie_ids.astype(np.int32))
                inputs[2].set_data_from_numpy(user_features.astype(np.float32))
                inputs[3].set_data_from_numpy(item_features.astype(np.float32))

                outputs = [httpclient.InferRequestedOutput("output_1")]
                response = self.triton_client.infer(model_name=self.model_name, inputs=inputs, outputs=outputs)
                return response.as_numpy("output_1").reshape(-1)
            except Exception as e:
                logger.warning("Triton inference failed, falling back to local model: %s", e)
                # Fall back to local below

        # 2. In-process local SavedModel execution
        if self.local_model is not None:
            import tensorflow as tf

            inputs_dict = {
                "userId": tf.constant(user_ids, dtype=tf.int32),
                "movieId": tf.constant(movie_ids, dtype=tf.int32),
                "user_features": tf.constant(user_features, dtype=tf.float32),
                "item_features": tf.constant(item_features, dtype=tf.float32),
            }
            # Handle standard serving signature
            if hasattr(self.local_model, "signatures") and "serving_default" in self.local_model.signatures:
                infer_fn = self.local_model.signatures["serving_default"]
                preds = infer_fn(**inputs_dict)
                out_tensor = preds.get("output_1", list(preds.values())[0])
                return out_tensor.numpy().reshape(-1)
            else:
                preds = self.local_model(inputs_dict)
                return preds.numpy().reshape(-1)

        raise RuntimeError("Neither Triton nor local ranker model is available for inference.")
