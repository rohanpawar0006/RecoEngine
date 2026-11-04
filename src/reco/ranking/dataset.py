"""
Dataset generator for the Deep Ranking Model (DCN-V2).
Builds ranking datasets combining observed positives (y=1) and
hard negatives sampled from retrieval candidate lists (y=0),
enriched with user and item features from Phase 2.
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from src.reco.data.vocab import IdVocabulary
from src.reco.indexing.faiss_index import FaissIndex
from src.reco.utils.logging import get_logger

logger = get_logger(__name__)


def load_feature_lookups(
    user_features_path: str,
    item_features_path: str,
    user_vocab: IdVocabulary,
    movie_vocab: IdVocabulary,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build aligned dense numpy feature matrices indexed by contiguous user and movie indices.

    Returns:
        user_feat_matrix: shape (num_users, 2) [log1p_count, avg_rating]
        item_feat_matrix: shape (num_items, 3) [log1p_popularity, avg_rating, normalized_year]
    """
    user_df = pd.read_parquet(user_features_path).set_index("userId")
    item_df = pd.read_parquet(item_features_path).set_index("movieId")

    num_users = len(user_vocab)
    num_items = len(movie_vocab)

    user_mat = np.zeros((num_users, 2), dtype=np.float32)
    for raw_u, idx in user_vocab.id_to_index.items():
        if raw_u in user_df.index:
            row = user_df.loc[raw_u]
            user_mat[idx, 0] = np.log1p(float(row["user_interaction_count"]))
            user_mat[idx, 1] = float(row["user_avg_rating"])

    item_mat = np.zeros((num_items, 3), dtype=np.float32)
    for raw_m, idx in movie_vocab.id_to_index.items():
        if raw_m in item_df.index:
            row = item_df.loc[raw_m]
            item_mat[idx, 0] = np.log1p(float(row["item_popularity"]))
            item_mat[idx, 1] = float(row["item_avg_rating"])
            year = float(row["release_year"])
            item_mat[idx, 2] = (year - 1920.0) / 100.0 if year > 1900 else 0.0

    return user_mat, item_mat


def build_ranking_pairs(
    train_parquet_path: str,
    user_vocab: IdVocabulary,
    movie_vocab: IdVocabulary,
    faiss_index: FaissIndex,
    user_tower_model,
    negatives_per_positive: int = 2,
    max_users: int = 5000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate (userId_idx, movieId_idx, label) triples with hard negatives from retrieval.

    Returns:
        user_indices, movie_indices, labels
    """
    logger.info("Building ranking dataset with hard negatives (max_users=%d)...", max_users)
    rng = np.random.default_rng(seed)

    train_df = pd.read_parquet(train_parquet_path, columns=["userId", "movieId", "is_positive"])
    positives = train_df[train_df["is_positive"] == 1]

    # Seen items per user in training
    user_seen = train_df.groupby("userId")["movieId"].apply(set).to_dict()
    user_positives = positives.groupby("userId")["movieId"].apply(list).to_dict()

    eligible_users = [u for u in user_positives if u in user_vocab.id_to_index]
    if len(eligible_users) > max_users:
        selected_users = list(rng.choice(eligible_users, size=max_users, replace=False))
    else:
        selected_users = eligible_users

    # Compute user embeddings for hard negative candidate generation
    user_idx_arr = np.array([user_vocab.transform(u) for u in selected_users], dtype=np.int32)
    user_embs = user_tower_model(tf.constant(user_idx_arr)).numpy()

    cand_item_ids, _ = faiss_index.search_batch(user_embs, k=100)

    u_list = []
    m_list = []
    y_list = []

    for i, raw_u in enumerate(selected_users):
        u_idx = user_idx_arr[i]
        user_pos_items = user_positives[raw_u]
        seen = user_seen[raw_u]

        # 1. Add positives (label = 1)
        # Cap positives per user to keep balance
        sampled_pos = user_pos_items[:min(len(user_pos_items), 20)]
        for mid in sampled_pos:
            m_idx = movie_vocab.transform(mid)
            if m_idx >= 0:
                u_list.append(u_idx)
                m_list.append(m_idx)
                y_list.append(1.0)

        # 2. Add hard negatives from retrieval candidates (label = 0)
        cands = cand_item_ids[i]
        hard_negs = [cid for cid in cands if cid not in seen]
        num_negs_needed = len(sampled_pos) * negatives_per_positive
        selected_negs = hard_negs[:num_negs_needed]

        for mid in selected_negs:
            m_idx = movie_vocab.transform(mid)
            if m_idx >= 0:
                u_list.append(u_idx)
                m_list.append(m_idx)
                y_list.append(0.0)

    u_arr = np.array(u_list, dtype=np.int32)
    m_arr = np.array(m_list, dtype=np.int32)
    y_arr = np.array(y_list, dtype=np.float32)

    logger.info(
        "Ranking dataset built: %d total samples (%d positives, %d hard negatives).",
        len(y_arr),
        int(np.sum(y_arr == 1.0)),
        int(np.sum(y_arr == 0.0)),
    )

    return u_arr, m_arr, y_arr


def create_tf_ranking_dataset(
    user_indices: np.ndarray,
    movie_indices: np.ndarray,
    labels: np.ndarray,
    user_feat_matrix: np.ndarray,
    item_feat_matrix: np.ndarray,
    batch_size: int = 2048,
    shuffle: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    """Pack ranking samples into a fast batched tf.data.Dataset."""
    u_feats = user_feat_matrix[user_indices]
    i_feats = item_feat_matrix[movie_indices]

    dataset = tf.data.Dataset.from_tensor_slices((
        {
            "userId": user_indices,
            "movieId": movie_indices,
            "user_features": u_feats,
            "item_features": i_feats,
        },
        labels,
    ))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=50000, seed=seed)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
