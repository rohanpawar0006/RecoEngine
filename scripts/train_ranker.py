"""
Training and Evaluation Pipeline for Deep Ranking Model (DCN-V2).
1. Loads feature tables, vocabularies, and Stage 1 retrieval artifacts (user tower & FAISS index).
2. Generates ranking training dataset with hard negatives sampled from retrieval candidate lists.
3. Trains DCN-V2 ranker with early stopping.
4. Evaluates end-to-end two-stage re-ranking: Retrieval-only vs. Retrieval + Ranker on NDCG@10.
5. Exports ranker SavedModel to artifacts/v1/ranker/ and saves reports/ranking_metrics.json.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Set up project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf
import tf_keras as keras

from src.reco.data.vocab import IdVocabulary
from src.reco.evaluation.metrics import evaluate_recommendations
from src.reco.indexing.faiss_index import FaissIndex
from src.reco.ranking.dataset import (
    build_ranking_pairs,
    create_tf_ranking_dataset,
    load_feature_lookups,
)
from src.reco.ranking.model import DCNv2Ranker
from src.reco.utils.logging import get_logger
from src.reco.utils.seeds import set_seed

logger = get_logger("train_ranker")


def train_and_evaluate_ranker(
    version: str = "v1",
    epochs: int = 5,
    batch_size: int = 2048,
    learning_rate: float = 0.0005,
    max_train_users: int = 8000,
    eval_users_count: int = 2000,
) -> dict:
    """Train DCN-V2 ranker and evaluate two-stage re-ranking on validation interactions."""
    config_path = os.path.join(PROJECT_ROOT, "configs", "ranking.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            seed = cfg.get("seed", 42)
    else:
        seed = 42

    set_seed(seed)

    # 1. Resolve paths
    proc_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    feat_dir = os.path.join(PROJECT_ROOT, "data", "features")
    artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts", version)
    reports_dir = os.path.join(PROJECT_ROOT, "reports")

    user_vocab = IdVocabulary.load(os.path.join(proc_dir, "vocab_users.json"))
    movie_vocab = IdVocabulary.load(os.path.join(proc_dir, "vocab_movies.json"))

    faiss_index_path = os.path.join(artifacts_dir, "faiss.index")
    item_ids_path = os.path.join(artifacts_dir, "item_ids.npy")
    user_tower_path = os.path.join(artifacts_dir, "user_tower")

    logger.info("Loading Stage 1 Retrieval artifacts from %s ...", artifacts_dir)
    faiss_index = FaissIndex.load(faiss_index_path, item_ids_path, embedding_dim=64)
    user_tower = keras.models.load_model(user_tower_path)

    # 2. Load aligned feature lookup matrices
    logger.info("Loading user and item features from %s ...", feat_dir)
    user_feat_mat, item_feat_mat = load_feature_lookups(
        user_features_path=os.path.join(feat_dir, "user_features.parquet"),
        item_features_path=os.path.join(feat_dir, "item_features.parquet"),
        user_vocab=user_vocab,
        movie_vocab=movie_vocab,
    )

    # 3. Build ranking dataset with hard negatives from Stage 1 retrieval
    train_parquet = os.path.join(proc_dir, "train.parquet")
    u_arr, m_arr, y_arr = build_ranking_pairs(
        train_parquet_path=train_parquet,
        user_vocab=user_vocab,
        movie_vocab=movie_vocab,
        faiss_index=faiss_index,
        user_tower_model=user_tower,
        negatives_per_positive=2,
        max_users=max_train_users,
        seed=seed,
    )

    # Split ranking samples into train (85%) and val (15%) for model training
    n_samples = len(y_arr)
    perm = np.random.default_rng(seed).permutation(n_samples)
    split_idx = int(n_samples * 0.85)

    train_idx = perm[:split_idx]
    val_idx = perm[split_idx:]

    train_ds = create_tf_ranking_dataset(
        user_indices=u_arr[train_idx],
        movie_indices=m_arr[train_idx],
        labels=y_arr[train_idx],
        user_feat_matrix=user_feat_mat,
        item_feat_matrix=item_feat_mat,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    val_ds = create_tf_ranking_dataset(
        user_indices=u_arr[val_idx],
        movie_indices=m_arr[val_idx],
        labels=y_arr[val_idx],
        user_feat_matrix=user_feat_mat,
        item_feat_matrix=item_feat_mat,
        batch_size=batch_size,
        shuffle=False,
    )

    # 4. Initialize DCN-V2 Ranker Model
    logger.info("Initializing DCN-V2 Ranking Model...")
    ranker = DCNv2Ranker(
        num_users=len(user_vocab),
        num_items=len(movie_vocab),
        embedding_dim=32,
        num_cross_layers=2,
        deep_units=[128, 64, 32],
        dropout_rate=0.2,
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    ranker.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.AUC(name="auc")],
    )

    # 5. Train Ranker with Early Stopping
    logger.info("Training DCN-V2 Ranker for up to %d epochs...", epochs)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    ranker.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # 6. Two-Stage Candidate Re-Ranking Evaluation on Validation Interactions
    logger.info("Evaluating Two-Stage Pipeline (Retrieval vs. Retrieval + Ranker)...")
    val_df = pd.read_parquet(os.path.join(proc_dir, "val.parquet"))
    val_pos = val_df[val_df["is_positive"] == 1]

    # Warm validation users
    val_known = val_pos[val_pos["userId"].isin(user_vocab.id_to_index)]
    val_gt = val_known.groupby("userId")["movieId"].apply(set).to_dict()

    train_df = pd.read_parquet(train_parquet, columns=["userId", "movieId"])
    train_seen_by_user = train_df.groupby("userId")["movieId"].apply(set).to_dict()

    eval_users = list(val_gt.keys())
    if len(eval_users) > eval_users_count:
        rng = np.random.default_rng(seed)
        eval_users = list(rng.choice(eval_users, size=eval_users_count, replace=False))

    logger.info("Re-ranking candidates for %d warm validation users...", len(eval_users))

    retrieval_only_recs = {}
    reranked_recs = {}

    eval_user_indices = np.array([user_vocab.transform(u) for u in eval_users], dtype=np.int32)
    user_embs = user_tower(tf.constant(eval_user_indices)).numpy()

    # Retrieve top 200 candidates per user via FAISS
    cand_matrix, _ = faiss_index.search_batch(user_embs, k=250)

    for i, u in enumerate(eval_users):
        u_idx = eval_user_indices[i]
        seen = train_seen_by_user.get(u, set())
        raw_cands = [cid for cid in cand_matrix[i] if cid not in seen][:200]

        # Stage 1: Retrieval-only top-10
        retrieval_only_recs[u] = raw_cands[:10]

        # Stage 2: Re-rank all 200 candidates using DCN-V2 Ranker
        cand_m_indices = np.array([movie_vocab.transform(mid) for mid in raw_cands], dtype=np.int32)
        n_cands = len(cand_m_indices)

        u_idx_repeated = np.full((n_cands,), u_idx, dtype=np.int32)
        u_feat_repeated = np.repeat(user_feat_mat[u_idx : u_idx + 1], n_cands, axis=0)
        i_feat_batch = item_feat_mat[cand_m_indices]

        batch_inputs = {
            "userId": tf.constant(u_idx_repeated),
            "movieId": tf.constant(cand_m_indices),
            "user_features": tf.constant(u_feat_repeated),
            "item_features": tf.constant(i_feat_batch),
        }

        # Predict CTR scores
        pred_scores = ranker(batch_inputs, training=False).numpy()
        ranked_order = np.argsort(-pred_scores)
        reranked_movie_ids = [raw_cands[idx] for idx in ranked_order]
        reranked_recs[u] = reranked_movie_ids[:10]

    # Evaluate NDCG@10 and Recall@10 for both stages on exact same ground truth
    eval_gt_dict = {u: val_gt[u] for u in eval_users}
    retrieval_metrics = evaluate_recommendations(retrieval_only_recs, eval_gt_dict, k_list=[10])
    reranked_metrics = evaluate_recommendations(reranked_recs, eval_gt_dict, k_list=[10])

    r_ndcg_10 = retrieval_metrics["ndcg@10"]
    rerank_ndcg_10 = reranked_metrics["ndcg@10"]
    beats_retrieval = bool(rerank_ndcg_10 > r_ndcg_10)

    logger.info("=== Two-Stage Evaluation Results (Top-10) ===")
    logger.info("Stage 1: Retrieval-Only   - NDCG@10: %.5f, Recall@10: %.5f", r_ndcg_10, retrieval_metrics["recall@10"])
    logger.info("Stage 2: Retrieval+Ranker - NDCG@10: %.5f, Recall@10: %.5f", rerank_ndcg_10, reranked_metrics["recall@10"])
    logger.info(
        "Hurdle check: Stage 2 NDCG@10 = %.5f vs Stage 1 = %.5f (Beats retrieval: %s, Margin: +%.5f)",
        rerank_ndcg_10,
        r_ndcg_10,
        beats_retrieval,
        rerank_ndcg_10 - r_ndcg_10,
    )

    # 7. Save Artifacts & Reports
    ranker_export_path = os.path.join(artifacts_dir, "ranker")
    logger.info("Saving ranker SavedModel to %s ...", ranker_export_path)
    ranker.save(ranker_export_path, save_format="tf")

    ranking_report = {
        "version": version,
        "evaluation_users_count": len(eval_users),
        "stage_1_retrieval_only": retrieval_metrics,
        "stage_2_retrieval_plus_ranker": reranked_metrics,
        "beats_retrieval_hurdle": beats_retrieval,
        "ndcg@10_margin": float(rerank_ndcg_10 - r_ndcg_10),
        "relative_improvement_pct": float((rerank_ndcg_10 - r_ndcg_10) / r_ndcg_10 * 100) if r_ndcg_10 > 0 else 0.0,
    }

    report_path = os.path.join(reports_dir, "ranking_metrics.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(ranking_report, f, indent=2)

    logger.info("Saved ranking metrics to %s", report_path)
    return ranking_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deep Ranking Model (DCN-V2)")
    parser.add_argument("--version", type=str, default="v1", help="Artifact version")
    parser.add_argument("--epochs", type=int, default=5, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--max-users", type=int, default=8000, help="Max users for training dataset")
    parser.add_argument("--eval-users", type=int, default=2000, help="Users for validation evaluation")
    args = parser.parse_args()

    train_and_evaluate_ranker(
        version=args.version,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_train_users=args.max_users,
        eval_users_count=args.eval_users,
    )
