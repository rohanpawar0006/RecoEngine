"""
FAISS Nearest Neighbor Indexing and Search Wrapper.
Wraps faiss-cpu for fast candidate retrieval over item embeddings.
Supports exact inner product (IndexFlatIP), seen-item filtering, and serialization.
"""

import os

import faiss
import numpy as np

from src.reco.utils.logging import get_logger

logger = get_logger(__name__)


class FaissIndex:
    """Exact and approximate nearest neighbor index over catalog embeddings."""

    def __init__(self, embedding_dim: int = 64, index_type: str = "FlatIP") -> None:
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.item_ids: np.ndarray | None = None

        if index_type == "FlatIP":
            self.index = faiss.IndexFlatIP(embedding_dim)
        else:
            raise ValueError(f"Unsupported index type: {index_type}. Use 'FlatIP'.")

    @property
    def ntotal(self) -> int:
        """Return total number of indexed vectors."""
        return self.index.ntotal

    def add(self, embeddings: np.ndarray, item_ids: np.ndarray) -> None:
        """
        Add item embeddings to the index.

        Args:
            embeddings: Float32 matrix of shape (N, embedding_dim).
            item_ids: Array of raw item IDs corresponding to rows.
        """
        assert embeddings.shape[1] == self.embedding_dim, (
            f"Embedding dim mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}"
        )
        assert len(embeddings) == len(item_ids), "Embeddings and item_ids count mismatch."

        embeddings_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(embeddings_f32)
        self.item_ids = np.array(item_ids, dtype=np.int32)
        logger.info("Added %d vectors to FAISS index. Total: %d", len(embeddings), self.ntotal)

    def search_single(
        self,
        query_vector: np.ndarray,
        k: int = 200,
        seen_items: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """
        Search top-k nearest items for a single user embedding.

        Args:
            query_vector: 1D array of shape (embedding_dim,) or 2D (1, embedding_dim).
            k: Number of candidates to return.
            seen_items: Set of item IDs to exclude from results.

        Returns:
            List of (movieId, score) tuples ordered by descending score.
        """
        query = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        fetch_k = min(self.ntotal, k + (len(seen_items) if seen_items else 0) + 50)

        scores, indices = self.index.search(query, fetch_k)
        scores = scores[0]
        indices = indices[0]

        results: list[tuple[int, float]] = []
        for idx, score in zip(indices, scores):
            if idx < 0:
                continue
            item_id = int(self.item_ids[idx])
            if seen_items and item_id in seen_items:
                continue
            results.append((item_id, float(score)))
            if len(results) == k:
                break

        return results

    def search_batch(
        self,
        query_vectors: np.ndarray,
        k: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Batch search without seen-item filtering (returns raw IDs and scores).

        Args:
            query_vectors: 2D array of shape (B, embedding_dim).
            k: Number of nearest neighbors per query.

        Returns:
            Tuple of (candidate_item_ids, candidate_scores) each of shape (B, k).
        """
        queries = np.ascontiguousarray(query_vectors, dtype=np.float32)
        scores, indices = self.index.search(queries, k)
        item_ids = self.item_ids[indices]
        return item_ids, scores

    def save(self, index_path: str, ids_path: str) -> None:
        """Save FAISS index and item IDs to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
        faiss.write_index(self.index, index_path)
        np.save(ids_path, self.item_ids)
        logger.info("Saved FAISS index to %s and IDs to %s", index_path, ids_path)

    @classmethod
    def load(cls, index_path: str, ids_path: str, embedding_dim: int = 64) -> "FaissIndex":
        """Load FAISS index and item IDs from disk."""
        instance = cls(embedding_dim=embedding_dim, index_type="FlatIP")
        instance.index = faiss.read_index(str(index_path))
        instance.item_ids = np.load(str(ids_path))
        logger.info("Loaded FAISS index (%d vectors) from %s", instance.ntotal, index_path)
        return instance
