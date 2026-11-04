"""
Training and Evaluation Pipeline for Two-Tower Retrieval Model.
1. Loads training interactions and ID vocabularies.
2. Initializes and trains TwoTowerModel with in-batch negatives.
3. Computes catalog item embeddings matrix.
4. Evaluates validation Recall@10, Recall@50, Recall@200, and NDCG@10.
5. Exports user tower SavedModel and item embeddings to artifacts/v1/.
6. Generates reports/retrieval_metrics.json.
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
from src.reco.retrieval.dataset import create_retrieval_dataset, get_catalog_indices
from src.reco.retrieval.model import TwoTowerModel
from src.reco.utils.logging import get_logger
from src.reco.utils.seeds import set_seed

logger = get_logger("train_retrieval")


def train_and_evaluate(
    data_mode: str = "dev",
    epochs: int = 5,
    batch_size: int = 4096,
    learning_rate: float = 0.001,
    embedding_dim: int = 64,
    max_samples: int | None = None,
    version: str = "v1",
) -> dict:
    """Train two-tower model and evaluate retrieval recall and NDCG."""
    config_path = os.path.join(PROJECT_ROOT, "configs", "retrieval.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            seed = cfg.get("seed", 42)
    else:
        seed = 42

    set_seed(seed)

    # 1. Resolve paths
    proc_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts", version)
    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    user_vocab_path = os.path.join(proc_dir, "vocab_users.json")
    movie_vocab_path = os.path.join(proc_dir, "vocab_movies.json")

    user_vocab = IdVocabulary.load(user_vocab_path)
    movie_vocab = IdVocabulary.load(movie_vocab_path)

    if data_mode == "dev":
        train_parquet = os.path.join(proc_dir, "dev", "train.parquet")
        val_parquet = os.path.join(proc_dir, "dev", "val.parquet")
    else:
        train_parquet = os.path.join(proc_dir, "train.parquet")
        val_parquet = os.path.join(proc_dir, "val.parquet")

    logger.info("Starting Two-Tower Retrieval Training (data_mode=%s, epochs=%d)", data_mode, epochs)
    logger.info("Vocabulary sizes: %d users, %d movies", len(user_vocab), len(movie_vocab))

    # 2. Build training dataset
    train_ds = create_retrieval_dataset(
        interactions_path=train_parquet,
        user_vocab=user_vocab,
        movie_vocab=movie_vocab,
        batch_size=batch_size,
        positive_only=True,
        max_samples=max_samples,
        seed=seed,
    )

    # 3. Initialize Model and Optimizer
    model = TwoTowerModel(
        num_users=len(user_vocab),
        num_items=len(movie_vocab),
        embedding_dim=embedding_dim,
        temperature=0.05,
    )
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer)

    # 4. Training Loop
    logger.info("Training two-tower model...")
    for epoch in range(1, epochs + 1):
        epoch_losses = []
        for batch in train_ds:
            loss_dict = model.train_step(batch)
            loss_val = float(loss_dict["loss"].numpy())
            epoch_losses.append(loss_val)
        mean_loss = float(np.mean(epoch_losses))
        logger.info("Epoch %d/%d - Average In-Batch Retrieval Loss: %.4f", epoch, epochs, mean_loss)

    # 5. Extract Full Catalog Item Embeddings Matrix
    logger.info("Computing catalog item embeddings...")
    catalog_indices = get_catalog_indices(movie_vocab)
    catalog_ds = tf.data.Dataset.from_tensor_slices(catalog_indices).batch(4096)

    item_embeddings_list = []
    for batch_items in catalog_ds:
        emb = model.item_tower(batch_items).numpy()
        item_embeddings_list.append(emb)

    item_embeddings = np.vstack(item_embeddings_list).astype(np.float32)
    item_raw_ids = np.array([movie_vocab.inverse_transform(idx) for idx in catalog_indices], dtype=np.int32)

    logger.info("Item embeddings matrix shape: %s", item_embeddings.shape)

    # 6. Evaluate on Validation Set
    logger.info("Evaluating retrieval Recall and NDCG on validation split...")
    val_df = pd.read_parquet(val_parquet)
    val_positives = val_df[val_df["is_positive"] == 1]

    # Only warm users (known to the training vocabulary) are evaluated for two-tower retrieval
    val_known_positives = val_positives[val_positives["userId"].isin(user_vocab.id_to_index)]
    val_ground_truth = val_known_positives.groupby("userId")["movieId"].apply(set).to_dict()

    train_df = pd.read_parquet(train_parquet, columns=["userId", "movieId"])
    train_seen_by_user = train_df.groupby("userId")["movieId"].apply(set).to_dict()

    eval_users = list(val_ground_truth.keys())
    if len(eval_users) > 5000:
        rng = np.random.default_rng(seed)
        eval_users = list(rng.choice(eval_users, size=5000, replace=False))

    logger.info("Evaluating on %d warm validation users...", len(eval_users))

    eval_user_indices = np.array([user_vocab.transform(u) for u in eval_users], dtype=np.int32)
    user_embeddings = model.user_tower(tf.constant(eval_user_indices)).numpy()

    # Load item popularity feature to apply calibrated frequency prior
    feat_parquet = os.path.join(PROJECT_ROOT, "data", "features", "item_features.parquet")
    if os.path.exists(feat_parquet):
        item_feats = pd.read_parquet(feat_parquet).set_index("movieId")
        pop_counts = np.array(
            [item_feats.loc[mid, "item_popularity"] if mid in item_feats.index else 0 for mid in item_raw_ids]
        )
        log_pop = np.log1p(pop_counts).astype(np.float32)
        log_pop_norm = log_pop / (log_pop.max() + 1e-6)
    else:
        log_pop_norm = 0.0

    # Two-tower similarity score combined with popularity prior
    similarity_scores = np.matmul(user_embeddings, item_embeddings.T) + 0.4 * log_pop_norm

    recommendations = {}
    top_k_retrieve = 300  # retrieve extra to filter train-seen items

    for i, u in enumerate(eval_users):
        seen = train_seen_by_user.get(u, set())
        cand_indices = np.argpartition(-similarity_scores[i], top_k_retrieve)[:top_k_retrieve]
        cand_sorted_indices = cand_indices[np.argsort(-similarity_scores[i, cand_indices])]
        cand_movie_ids = item_raw_ids[cand_sorted_indices]

        # Filter seen items
        filtered = [mid for mid in cand_movie_ids if mid not in seen][:200]
        recommendations[u] = filtered

    eval_gt = {u: val_ground_truth[u] for u in eval_users}
    retrieval_metrics = evaluate_recommendations(
        recommendations=recommendations,
        ground_truth=eval_gt,
        k_list=[10, 50, 200],
    )

    # 7. Compare against Popularity Baseline on the Exact Same Warm Users
    from src.reco.evaluation.baselines import PopularityBaseline

    pop_baseline = PopularityBaseline(train_df=train_df, positive_only=True)
    pop_recs = {
        u: pop_baseline.recommend(u, k=200, seen_items=train_seen_by_user.get(u, set()))
        for u in eval_users
    }
    pop_warm_metrics = evaluate_recommendations(
        recommendations=pop_recs,
        ground_truth=eval_gt,
        k_list=[10, 50, 200],
    )

    pop_warm_recall_200 = pop_warm_metrics.get("recall@200", 0.0)
    retrieval_recall_200 = retrieval_metrics.get("recall@200", 0.0)
    beats_popularity = bool(retrieval_recall_200 > pop_warm_recall_200)

    logger.info("Two-Tower Retrieval Validation Results (Warm Users):")
    logger.info(json.dumps(retrieval_metrics, indent=2))
    logger.info("Popularity Baseline Validation Results (Warm Users):")
    logger.info(json.dumps(pop_warm_metrics, indent=2))

    # 8. Save Artifacts & Reports
    logger.info("Saving model artifacts to %s ...", artifacts_dir)

    # Save user tower as SavedModel
    user_tower_dir = os.path.join(artifacts_dir, "user_tower")
    model.user_tower.save(user_tower_dir, save_format="tf")

    # Save item embeddings matrix and ID mapping
    np.save(os.path.join(artifacts_dir, "item_embeddings.npy"), item_embeddings)
    np.save(os.path.join(artifacts_dir, "item_ids.npy"), item_raw_ids)

    # Save hyperparameters
    hyperparams = {
        "version": version,
        "data_mode": data_mode,
        "embedding_dim": embedding_dim,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "seed": seed,
        "temperature": 0.05,
        "popularity_prior_weight": 0.4,
        "num_users": len(user_vocab),
        "num_items": len(movie_vocab),
    }
    with open(os.path.join(artifacts_dir, "hyperparams.json"), "w", encoding="utf-8") as f:
        json.dump(hyperparams, f, indent=2)

    # Save retrieval metrics report
    report = {
        "version": version,
        "data_mode": data_mode,
        "evaluation_users_count": len(eval_users),
        "retrieval_metrics": retrieval_metrics,
        "popularity_baseline_warm_metrics": pop_warm_metrics,
        "beats_popularity_hurdle": beats_popularity,
        "recall@200_margin": float(retrieval_recall_200 - pop_warm_recall_200),
    }
    report_path = os.path.join(reports_dir, "retrieval_metrics.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved retrieval metrics to %s", report_path)
    logger.info(
        "Hurdle check (Warm Users): Retrieval Recall@200 = %.5f vs Popularity = %.5f (Beats baseline: %s, Margin: +%.5f)",
        retrieval_recall_200,
        pop_warm_recall_200,
        beats_popularity,
        retrieval_recall_200 - pop_warm_recall_200,
    )

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Two-Tower Retrieval Model")
    parser.add_argument("--data", choices=["dev", "full"], default="dev", help="Dataset partition (dev or full)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4096, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--max-samples", type=int, default=None, help="Max recent training samples to use")
    parser.add_argument("--version", type=str, default="v1", help="Artifact version")
    args = parser.parse_args()

    train_and_evaluate(
        data_mode=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        embedding_dim=args.dim,
        max_samples=args.max_samples,
        version=args.version,
    )
