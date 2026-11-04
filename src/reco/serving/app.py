"""
FastAPI Serving Application for RecoEngine.
Orchestrates: Cache -> User Tower -> FAISS Retrieval -> DCN-V2 Ranker -> Seen Filter -> Cold Fallbacks.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.reco.fallback.content_based import ContentBasedRecommender  # noqa: E402
from src.reco.fallback.popularity import TimeDecayedPopularity  # noqa: E402
from src.reco.indexing.faiss_index import FaissIndex  # noqa: E402
from src.reco.serving.ab import get_experiment_variant  # noqa: E402
from src.reco.serving.cache import RedisCache  # noqa: E402
from src.reco.serving.inference import RankerInference  # noqa: E402
from src.reco.serving.schemas import (  # noqa: E402
    ABSummaryResponse,
    EventRequest,
    EventResponse,
    HealthResponse,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)

logger = logging.getLogger("reco.serving")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

# Global serving state
state: Dict[str, Any] = {}


def load_serving_state():
    """Load all offline models, indexes, vocabularies, and feature tables into memory."""
    logger.info("Initializing RecoEngine serving dependencies...")
    state["version"] = "v1"

    # 1. Movies metadata
    movies_path = PROJECT_ROOT / "data" / "raw" / "ml-25m" / "movies.csv"
    if movies_path.exists():
        movies_df = pd.read_csv(movies_path)
        movie_meta = {}
        for _, r in movies_df.iterrows():
            movie_meta[int(r["movieId"])] = {
                "title": str(r["title"]),
                "genres": str(r["genres"]),
            }
        state["movie_meta"] = movie_meta
    else:
        state["movie_meta"] = {}

    # 2. Vocabularies
    vocab_users_path = PROJECT_ROOT / "data" / "processed" / "vocab_users.json"
    if vocab_users_path.exists():
        with open(vocab_users_path, "r", encoding="utf-8") as f:
            v_data = json.load(f)
            state["vocab_users"] = {int(k): int(v) for k, v in v_data.get("id_to_index", {}).items()}
    else:
        state["vocab_users"] = {}

    vocab_movies_path = PROJECT_ROOT / "data" / "processed" / "vocab_movies.json"
    if vocab_movies_path.exists():
        with open(vocab_movies_path, "r", encoding="utf-8") as f:
            v_data = json.load(f)
            state["vocab_movies"] = {int(k): int(v) for k, v in v_data.get("id_to_index", {}).items()}
    else:
        state["vocab_movies"] = {}

    # 3. Dense feature tables
    item_feat_path = PROJECT_ROOT / "data" / "features" / "item_features.parquet"
    if not item_feat_path.exists():
        item_feat_path = PROJECT_ROOT / "data" / "features" / "dev" / "item_features.parquet"

    item_feat_map = {}
    if item_feat_path.exists():
        idf = pd.read_parquet(item_feat_path)
        for _, r in idf.iterrows():
            year = float(r.get("release_year", 0.0))
            norm_year = (year - 1920.0) / 100.0 if year > 1900.0 else 0.0
            item_feat_map[int(r["movieId"])] = np.array(
                [
                    np.log1p(float(r.get("item_popularity", 0.0))),
                    float(r.get("item_avg_rating", 0.0)),
                    norm_year,
                ],
                dtype=np.float32,
            )
    state["item_features"] = item_feat_map

    user_feat_path = PROJECT_ROOT / "data" / "features" / "user_features.parquet"
    if not user_feat_path.exists():
        user_feat_path = PROJECT_ROOT / "data" / "features" / "dev" / "user_features.parquet"

    user_feat_map = {}
    user_seen_map = {}
    if user_feat_path.exists():
        udf = pd.read_parquet(user_feat_path)
        for _, r in udf.iterrows():
            u = int(r["userId"])
            user_feat_map[u] = np.array(
                [
                    np.log1p(float(r.get("user_interaction_count", 0.0))),
                    float(r.get("user_avg_rating", 0.0)),
                ],
                dtype=np.float32,
            )
            raw_seen = r.get("last_n_items", [])
            user_seen_map[u] = set(raw_seen.tolist() if hasattr(raw_seen, "tolist") else list(raw_seen))
    state["user_features"] = user_feat_map
    state["user_seen"] = user_seen_map

    # 4. User Tower SavedModel
    user_tower_path = PROJECT_ROOT / "artifacts" / "v1" / "user_tower"
    if user_tower_path.exists():
        try:
            import tensorflow as tf

            state["user_tower"] = tf.saved_model.load(str(user_tower_path))
            logger.info("Loaded User Tower SavedModel.")
        except Exception as e:
            logger.warning("Could not load User Tower: %s", e)
            state["user_tower"] = None
    else:
        state["user_tower"] = None

    # 5. FAISS Index & Item Embeddings
    faiss_path = PROJECT_ROOT / "artifacts" / "v1" / "faiss.index"
    emb_path = PROJECT_ROOT / "artifacts" / "v1" / "item_embeddings.npy"
    ids_path = PROJECT_ROOT / "artifacts" / "v1" / "item_ids.npy"

    if faiss_path.exists() and emb_path.exists() and ids_path.exists():
        index = FaissIndex.load(index_path=faiss_path, ids_path=ids_path)
        item_embs = np.load(emb_path)
        item_ids = np.load(ids_path)
        item_id_to_idx = {int(mid): i for i, mid in enumerate(item_ids)}

        state["faiss_index"] = index
        state["item_embeddings"] = item_embs
        state["item_ids"] = item_ids
        state["item_id_to_idx"] = item_id_to_idx
        logger.info("Loaded FAISS index (%d items).", len(item_ids))
    else:
        state["faiss_index"] = None
        state["item_embeddings"] = None
        state["item_ids"] = None
        state["item_id_to_idx"] = {}

    # 6. Fallback Models
    pop_path = PROJECT_ROOT / "artifacts" / "v1" / "fallback_popularity.json"
    if pop_path.exists():
        state["fallback_pop"] = TimeDecayedPopularity.load(pop_path)
    else:
        state["fallback_pop"] = TimeDecayedPopularity()

    content_path = PROJECT_ROOT / "artifacts" / "v1" / "fallback_content.json"
    if content_path.exists():
        state["fallback_content"] = ContentBasedRecommender.load(content_path)
    else:
        state["fallback_content"] = None

    # 7. Redis Cache
    state["cache"] = RedisCache(host="localhost", port=6379)

    # 8. Ranker Inference (Triton / Local Fallback)
    ranker_path = PROJECT_ROOT / "artifacts" / "v1" / "ranker"
    state["ranker"] = RankerInference(local_model_path=ranker_path)

    # 9. Events Log Path
    logs_dir = PROJECT_ROOT / "data" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    state["events_path"] = logs_dir / "events.jsonl"

    logger.info("Serving layer initialization complete. Served by: %s", state["ranker"].served_by)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_serving_state()
    yield


app = FastAPI(
    title="RecoEngine API",
    description="Production Two-Stage Recommendation Engine on MovieLens 25M (TFRS Two-Tower + FAISS + DCN-V2)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return health status of API, Redis, Triton/local ranker, and FAISS index."""
    cache: RedisCache = state.get("cache")
    ranker: RankerInference = state.get("ranker")
    faiss_idx: FaissIndex = state.get("faiss_index")

    redis_ok = cache.is_available() if cache else False
    triton_ok = ranker.is_triton_available() if ranker else False
    served_by = ranker.served_by if ranker else "none"
    faiss_ok = faiss_idx is not None
    catalog_size = len(state.get("item_ids", [])) if state.get("item_ids") is not None else 0

    return HealthResponse(
        status="healthy" if (faiss_ok and ranker and ranker.is_ready()) else "degraded",
        version=state.get("version", "v1"),
        redis_available=redis_ok,
        triton_available=triton_ok,
        served_by=served_by,
        faiss_loaded=faiss_ok,
        catalog_size=catalog_size,
    )


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(req: RecommendationRequest) -> RecommendationResponse:
    """
    Generate personalized recommendations for a user:
    1. Check Redis cache.
    2. If warm user: Two-tower user embedding -> FAISS top-200 -> DCN-V2 ranking -> top-K.
    3. If cold user: Time-decayed popularity or Content-based seed matching.
    4. Store result in Redis cache.
    """
    t0 = time.perf_counter()
    user_id = req.user_id
    k = req.k
    if req.variant in [None, "", "auto"]:
        variant = get_experiment_variant(user_id)
    else:
        variant = req.variant
    seed_genres = req.seed_genres or []

    cache: RedisCache = state.get("cache")

    # 1. Cache lookup
    if cache:
        cached_items = cache.get_recommendations(variant=variant, user_id=user_id, k=k)
        if cached_items is not None:
            t1 = time.perf_counter()
            items = [RecommendationItem(**item) for item in cached_items]
            return RecommendationResponse(
                user_id=user_id,
                variant=variant,
                model_version=state.get("version", "v1"),
                source="cache",
                cache_hit=True,
                latency_ms=round((t1 - t0) * 1000.0, 3),
                items=items,
            )

    # 2. Check if user is warm
    is_warm = (
        user_id in state.get("vocab_users", {})
        and user_id in state.get("user_features", {})
        and state.get("user_tower") is not None
        and state.get("faiss_index") is not None
    )

    items: List[RecommendationItem] = []
    source = "fallback_popularity"

    if is_warm:
        # --- Warm User: Two-Stage Model Pipeline ---
        user_idx = state["vocab_users"][user_id]
        seen_items = state["user_seen"].get(user_id, set())

        # Step A: User Tower Embedding
        import tensorflow as tf

        raw_user_emb = state["user_tower"](tf.constant([user_idx], dtype=tf.int32)).numpy().astype(np.float32)
        norm = np.linalg.norm(raw_user_emb, axis=1, keepdims=True)
        user_emb = raw_user_emb / np.maximum(norm, 1e-12)

        # Step B: FAISS Retrieval (Top 200)
        retrieved_pairs = state["faiss_index"].search_single(query_vector=user_emb, k=200, seen_items=seen_items)
        candidate_ids = [mid for mid, _ in retrieved_pairs]

        # Step C: Variant Decision (A = Retrieval-Only, B = Retrieval + DCN-V2 Ranker)
        movie_meta = state.get("movie_meta", {})
        if len(candidate_ids) > 0 and variant == "A":
            # Variant A: Pure Retrieval-Only Ordering (ordered by FAISS dot-product similarity)
            for rank_idx, (mid, faiss_score) in enumerate(retrieved_pairs[:k], start=1):
                mid_int = int(mid)
                meta = movie_meta.get(mid_int, {"title": f"Movie #{mid_int}", "genres": "Unknown"})
                items.append(
                    RecommendationItem(
                        item_id=mid_int,
                        title=meta["title"],
                        genres=meta["genres"],
                        score=round(float(faiss_score), 4),
                        rank=rank_idx,
                    )
                )
            source = "model_retrieval_only"
        elif len(candidate_ids) > 0 and state.get("ranker") and state["ranker"].is_ready():
            # Variant B / Default: Two-Stage Pipeline (DCN-V2 Ranker Re-scoring)
            u_feat = state["user_features"][user_id]
            user_features_batch = np.repeat(u_feat[np.newaxis, :], len(candidate_ids), axis=0)

            item_features_batch = np.array(
                [state["item_features"].get(mid, np.zeros(3, dtype=np.float32)) for mid in candidate_ids],
                dtype=np.float32,
            )
            user_ids_batch = np.full(len(candidate_ids), user_idx, dtype=np.int32)
            movie_ids_batch = np.array(
                [state["vocab_movies"].get(mid, 0) for mid in candidate_ids],
                dtype=np.int32,
            )

            scores = state["ranker"].score(
                user_ids=user_ids_batch,
                movie_ids=movie_ids_batch,
                user_features=user_features_batch,
                item_features=item_features_batch,
            )

            # Sort candidate IDs by ranker score descending
            ranked_indices = np.argsort(-scores)[:k]

            for rank_idx, idx in enumerate(ranked_indices, start=1):
                mid = int(candidate_ids[idx])
                meta = movie_meta.get(mid, {"title": f"Movie #{mid}", "genres": "Unknown"})
                items.append(
                    RecommendationItem(
                        item_id=mid,
                        title=meta["title"],
                        genres=meta["genres"],
                        score=round(float(scores[idx]), 4),
                        rank=rank_idx,
                    )
                )
            source = "model"
        else:
            # Fallback if candidates empty or ranker not ready
            is_warm = False

    if not is_warm:
        # --- Cold User: Fallback Pipeline ---
        movie_meta = state.get("movie_meta", {})

        if seed_genres and state.get("fallback_content"):
            chosen_ids = state["fallback_content"].recommend_from_seed_genres(seed_genres=seed_genres, k=k)
            source = "fallback_content"
        else:
            fallback_pop: TimeDecayedPopularity = state.get("fallback_pop")
            chosen_ids = fallback_pop.recommend_global(k=k) if fallback_pop else []
            source = "fallback_popularity"

        for rank_idx, mid in enumerate(chosen_ids[:k], start=1):
            meta = movie_meta.get(mid, {"title": f"Movie #{mid}", "genres": "Unknown"})
            items.append(
                RecommendationItem(
                    item_id=mid,
                    title=meta["title"],
                    genres=meta["genres"],
                    score=round(1.0 / rank_idx, 4),
                    rank=rank_idx,
                )
            )

    # 3. Store in Redis cache
    if cache and items:
        cache.set_recommendations(
            variant=variant,
            user_id=user_id,
            k=k,
            recommendations=[item.model_dump() for item in items],
        )

    t1 = time.perf_counter()
    return RecommendationResponse(
        user_id=user_id,
        variant=variant,
        model_version=state.get("version", "v1"),
        source=source,
        cache_hit=False,
        latency_ms=round((t1 - t0) * 1000.0, 3),
        items=items,
    )


@app.post("/events", response_model=EventResponse)
def log_event(evt: EventRequest) -> EventResponse:
    """Log user feedback (impression, click, rating) to append-only JSONL log."""
    events_path: Path = state.get("events_path")
    if not events_path:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Events path unconfigured")

    record = evt.model_dump()
    if record.get("ts") is None:
        record["ts"] = time.time()

    try:
        with open(events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return EventResponse(status="success", logged=True)
    except Exception as e:
        logger.error("Failed to append event: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to log event")


@app.get("/ab/summary", response_model=ABSummaryResponse)
def get_ab_summary() -> ABSummaryResponse:
    """Aggregate impressions, clicks, ratings, and simulated CTR per variant from events log."""
    events_path: Path = state.get("events_path")
    if not events_path or not events_path.exists():
        return ABSummaryResponse(total_events=0, variants={})

    counts: Dict[str, Dict[str, int]] = {}
    total = 0

    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                variant = str(rec.get("variant", "unknown"))
                evt_type = str(rec.get("event", "other"))

                if variant not in counts:
                    counts[variant] = {"impressions": 0, "clicks": 0, "ratings": 0}

                if evt_type == "impression":
                    counts[variant]["impressions"] += 1
                elif evt_type == "click":
                    counts[variant]["clicks"] += 1
                elif evt_type == "rating":
                    counts[variant]["ratings"] += 1

                total += 1
            except Exception:
                continue

    # Compute CTR
    variants_summary = {}
    for var, data in counts.items():
        imps = data["impressions"]
        clicks = data["clicks"]
        ctr = (clicks / imps) if imps > 0 else 0.0
        variants_summary[var] = {
            "impressions": imps,
            "clicks": clicks,
            "ratings": data["ratings"],
            "ctr": round(ctr, 4),
        }

    return ABSummaryResponse(total_events=total, variants=variants_summary)
