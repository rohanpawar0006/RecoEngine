"""
Unit tests for FAISS indexing, brute-force numerical equivalence, and save/load roundtrip.
"""

import os
import tempfile

import numpy as np
import pytest

from src.reco.indexing.faiss_index import FaissIndex


@pytest.fixture
def sample_catalog():
    """Synthetic normalized catalog embeddings and IDs."""
    np.random.seed(42)
    n_items = 200
    dim = 16
    raw_emb = np.random.randn(n_items, dim).astype(np.float32)
    # L2 normalize
    emb = raw_emb / np.linalg.norm(raw_emb, axis=1, keepdims=True)
    item_ids = np.arange(1000, 1000 + n_items, dtype=np.int32)
    return emb, item_ids, dim


def test_faiss_brute_force_equality(sample_catalog):
    """Verify FAISS IndexFlatIP matches brute-force NumPy dot product exact ranking."""
    emb, item_ids, dim = sample_catalog
    index = FaissIndex(embedding_dim=dim, index_type="FlatIP")
    index.add(emb, item_ids)

    np.random.seed(123)
    query = np.random.randn(dim).astype(np.float32)
    query = query / np.linalg.norm(query)

    # FAISS search
    faiss_results = index.search_single(query, k=10)
    faiss_ids = [r[0] for r in faiss_results]
    faiss_scores = [r[1] for r in faiss_results]

    # Brute-force numpy calculation
    brute_scores = np.dot(emb, query)
    brute_order = np.argsort(-brute_scores)[:10]
    expected_ids = item_ids[brute_order].tolist()
    expected_scores = brute_scores[brute_order].tolist()

    assert faiss_ids == expected_ids
    np.testing.assert_allclose(faiss_scores, expected_scores, atol=1e-5)


def test_faiss_save_load_roundtrip(sample_catalog):
    """Verify saving and reloading an index preserves all data and search results."""
    emb, item_ids, dim = sample_catalog
    index = FaissIndex(embedding_dim=dim, index_type="FlatIP")
    index.add(emb, item_ids)

    query = emb[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        index_file = os.path.join(tmpdir, "test.index")
        ids_file = os.path.join(tmpdir, "ids.npy")

        index.save(index_file, ids_file)
        loaded = FaissIndex.load(index_file, ids_file, embedding_dim=dim)

        orig_res = index.search_single(query, k=5)
        load_res = loaded.search_single(query, k=5)

        assert orig_res == load_res


def test_search_seen_item_filtering(sample_catalog):
    """Verify that seen items in user history are strictly filtered out."""
    emb, item_ids, dim = sample_catalog
    index = FaissIndex(embedding_dim=dim, index_type="FlatIP")
    index.add(emb, item_ids)

    query = emb[0]
    unfiltered = index.search_single(query, k=5)
    top_item = unfiltered[0][0]

    # Filter the top item
    filtered = index.search_single(query, k=5, seen_items={top_item})
    returned_ids = [r[0] for r in filtered]

    assert top_item not in returned_ids
    assert len(filtered) == 5
