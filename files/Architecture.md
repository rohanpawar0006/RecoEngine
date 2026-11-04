# Architecture — RecoEngine

**Version:** 1.0 · **Created:** 2026-11-04 · Implements `PRD.md`

Exact library versions are **TBD** and must be verified and pinned in Phase 0 (TensorFlow / Keras / TFRS compatibility in particular). Nothing here should be assumed to install on Python 3.13 without checking.

---

## 1. Architecture Overview

Two flows: an **offline training flow** (batch) and an **online serving flow** (per request).

```text
OFFLINE (Airflow-orchestrated)
Raw data → Clean/Split (Pandas/Spark) → Spark features → Train two-tower → Item embeddings → FAISS index
                                                     └→ Train ranker (hard negatives from retrieval)
                                         Evaluate → Gate → Export to Triton model repo → Flush Redis

ONLINE
User/Demo UI → FastAPI → Redis (cache hit? return)
                     └ miss → known user? 
                          yes → user tower (Triton) → FAISS top-200 → ranker (Triton) → top-K
                          no  → fallback (popularity / content-based)
                     → write Redis → return JSON + log event
```

Design principle: **retrieval is cheap and broad, ranking is expensive and narrow.** Redis protects both.

## 2. Technology Stack

| Layer | Technology | Purpose | Why | Alternatives considered |
|---|---|---|---|---|
| Language | Python (3.11/3.12 env if 3.13 is unsupported by TF) | All code | Required by ecosystem | — |
| Retrieval + ranking models | TensorFlow + TensorFlow Recommenders (Keras) | Two-tower, DCN-V2/Wide & Deep | Required by brief; TFRS has retrieval/ranking tasks and metrics | PyTorch (not in brief) |
| ANN search | FAISS (`faiss-cpu`) | Nearest-neighbor over item embeddings | Required by brief; fast, simple | ScaNN, Annoy |
| Big-data features | Apache Spark (PySpark, local mode) | Interaction tables, features | Required by brief; scales beyond RAM | Pandas (fine at 25M rows but not "large-scale") |
| Cache | Redis (Docker) + `redis-py` | Rec lists, popularity, hot features | Required by brief | In-process LRU |
| Orchestration | Apache Airflow (Docker) | Scheduled retraining DAG | Required by brief | cron, Prefect |
| Model serving | Triton Inference Server (Docker) | Serve TF SavedModels | Required by brief | TF Serving |
| API | FastAPI + Pydantic + Uvicorn | Orchestration layer, validation | Typed, auto OpenAPI docs (useful for API documentation deliverable) | Flask |
| Demo UI | Streamlit | Interactive demo | Fastest path to a good-looking demo | React (too costly for 14 days) |
| Data format | Parquet (PyArrow) | Intermediate tables | Columnar, Spark/Pandas compatible | CSV |
| Experiment tracking | JSON/CSV reports in `reports/` (+ optional MLflow, post-MVP) | Reproducible metrics | Keep MVP simple | MLflow, W&B |
| Testing | pytest | Unit/integration tests | Standard | — |
| Quality | ruff (lint+format), mypy (light) | Code quality | Fast | flake8, black |
| Containers | Docker Compose | Redis, Triton, Airflow, API | One-command local stack | Kubernetes (overkill) |
| Diagrams | Mermaid / draw.io exported PNG | Architecture diagram deliverable | Renders on GitHub | — |
| CI (optional) | GitHub Actions | Lint + tests | Cheap signal of quality | — |

## 3. System Components

| Component | Responsibility |
|---|---|
| `data` | Load, clean, ID-map, temporal split, write Parquet |
| `features` | Spark jobs for user/item features (training-window only) |
| `retrieval` | Two-tower model definition, training, tower export |
| `indexing` | Build/save/load FAISS index; search wrapper |
| `ranking` | Ranker definition, hard-negative sampling, training, export |
| `fallback` | Popularity lists, content-based similarity for cold start |
| `serving` | FastAPI app, Triton client, Redis cache layer, A/B bucketing, event logging |
| `evaluation` | Metrics (NDCG, Recall, simulated CTR), baselines, bootstrap CIs, latency benchmark |
| `airflow/dags` | Retraining DAG and promotion gate |
| `triton/model_repository` | Versioned model directories + `config.pbtxt` |
| `demo` | Streamlit UI |

## 4. Data Flow

**Training**
1. Raw ratings → keep rating ≥ 4 as positives (ratings retained as feature/label source) → map user/movie IDs to ints.
2. **Split by timestamp** (e.g., ~80/10/10 by time quantiles `[ASSUMPTION]`). Users whose first interaction falls after the train cutoff are *naturally cold-start* in val/test.
3. Spark computes features from the **train window only**.
4. Two-tower trained with in-batch sampled softmax; item embeddings computed for the full catalog; FAISS built.
5. For each training user, retrieve candidates; label held-in positives = 1, retrieved non-interacted = 0 (hard negatives) → ranker training set.
6. Evaluate on val (tuning) then test (final, run once per reported result).

**Serving (warm user)**: user ID → cached features → user tower → embedding → FAISS top-200 → batch to ranker → sort → drop already-seen items → top-K → cache → respond.

**Serving (cold user)**: unknown ID → fallback; optional seed genres/items → content-based; labeled `source=fallback_popularity|fallback_content`.

**Leakage rules (critical):** no feature, vocabulary statistic, popularity count, or normalizer may be computed using val/test rows. Item popularity used in serving is the train-window popularity.

## 5. API Architecture

Style: REST/JSON over HTTP, FastAPI, auto-generated OpenAPI at `/docs` (use as API documentation deliverable).

| Endpoint | Purpose |
|---|---|
| `GET /health` | Status of API, Redis, Triton, index loaded, model version |
| `POST /recommend` | Body `{user_id, k=10, variant=null, seed_genres=[]}` → recommendations |
| `POST /events` | Log impression/click `{user_id, item_id, variant, event, ts}` for A/B analysis |
| `GET /ab/summary` | Aggregated per-variant metrics from the event log |

**Response of `/recommend`:** `{user_id, variant, model_version, source, cache_hit, latency_ms, items:[{item_id, title, genres, score, rank}]}`.

**Validation:** Pydantic models; `k` bounded (1–100); `user_id` must be an int or string matching a pattern; reject oversized `seed_genres`.
**Errors:** structured `{error: {code, message}}` with correct HTTP status; never return stack traces; dependency failures degrade rather than 500 where a fallback exists.
**Auth:** none for local demo `[ASSUMPTION]` (out of scope); if exposed publicly, add an API key via env var.

## 6. Data Storage Architecture

No relational DB is required. `[ASSUMPTION]`

**Files (Parquet/NPY/FAISS):** `data/processed/{train,val,test}.parquet`, `vocab/*.json`, `features/*.parquet`, `artifacts/<version>/{user_tower,item_embeddings.npy,faiss.index,ranker}`.

**Redis keys** (all JSON or compact binary, all with TTL):

| Key | Content | TTL `[ASSUMPTION]` |
|---|---|---|
| `rec:{variant}:{user_id}:{k}` | final ranked list | 1 h |
| `pop:global` | popularity list | 24 h |
| `pop:genre:{genre}` | per-genre popularity | 24 h |
| `ufeat:{user_id}` | hot user feature vector | 6 h |

Keys include **model version** (`rec:v{N}:...`) so a new model never serves stale lists; cache is also flushed on promotion.

**Event log:** append-only JSONL (or SQLite if querying is needed) in `data/logs/events.jsonl`; fields: ts, user_id, variant, item_id, event, model_version.

**Entities:** User(id, history, features), Item(id, title, genres, year, features), Interaction(user, item, rating, ts), Event(...).
**Lifecycle:** processed data and artifacts are reproducible → can be regenerated; only raw dataset and code are the source of truth.

## 7. Folder Structure

```text
reco-engine/
├── README.md                 # setup, run, results, links
├── docs/                     # PRD, Architecture, Rules, Phases, Design, Memory, diagrams, API doc
├── configs/                  # YAML configs (paths, hyperparams, k, TTLs) – no secrets
├── data/                     # gitignored: raw/, processed/, features/, logs/
├── artifacts/                # gitignored: versioned models + indexes
├── src/reco/
│   ├── data/                 # load, clean, split
│   ├── features/             # Spark jobs
│   ├── retrieval/            # two-tower model + training
│   ├── indexing/             # FAISS build/search
│   ├── ranking/              # ranker + hard negatives + training
│   ├── fallback/             # popularity + content-based
│   ├── serving/              # FastAPI app, triton_client, cache, ab, events
│   ├── evaluation/           # metrics, baselines, ab_replay, latency
│   └── utils/                # config, logging, seeds, paths
├── airflow/
│   └── dags/                 # retrain_dag.py
├── triton/
│   └── model_repository/     # ranker/, user_tower/ with config.pbtxt
├── demo/                     # Streamlit app
├── docker/                   # Dockerfiles
├── docker-compose.yml
├── scripts/                  # CLI entry points per stage
├── tests/                    # unit + integration
├── notebooks/                # EDA only; no logic that the pipeline depends on
├── reports/                  # metrics tables, plots, latency, A/B results
├── requirements.txt          # pinned
└── .env.example              # placeholder variable names only
```

## 8. Security Architecture

- **Secrets:** env vars via `.env` (gitignored) + `.env.example`; never commit credentials (Airflow default admin password, Redis password if set).
- **Input validation:** Pydantic on every endpoint; bounds on `k`, list sizes, string lengths.
- **Redis/Triton exposure:** bind to localhost / Docker internal network; do not publish ports publicly.
- **Serialization:** do not `pickle.load` untrusted files; use NPY/JSON/SavedModel/FAISS formats.
- **Rate limiting:** simple per-IP limit in API only if publicly deployed (post-MVP).
- **Data protection:** public dataset, no PII; event logs contain only synthetic/dataset IDs.
- **Prompt injection:** N/A (no LLM in the system).
- **Confidentiality:** Nexvia-provided information must not be committed or published (see PRD Open Questions).

## 9. Scalability

- **Retrieval:** exact FAISS is fine at ~60k items; at millions switch to IVF-PQ/HNSW (post-MVP experiment showing recall/latency tradeoff).
- **Features:** Spark local mode here; same code runs on a cluster.
- **Serving:** stateless API → horizontal scaling behind a load balancer; Triton dynamic batching; Redis absorbs repeat traffic.
- **Training:** TF distributed strategies / GPU for larger data (documented, not built).
- **Embeddings refresh:** nightly/weekly via Airflow; user embeddings computed on request from fresh features.

## 10. Failure Points & Handling

| Failure | Detection | Handling |
|---|---|---|
| Redis unavailable | connection error / timeout | Skip cache, log warning, continue; `cache_hit=false` |
| Triton unavailable | health check/timeouts | Fall back to in-process model; flag `served_by=local` and log |
| FAISS index missing/corrupt | startup check | Fail fast at startup with clear error; `/health` red |
| Unknown user / empty history | ID not in vocab | Fallback recommendations |
| Candidate list too small after filtering seen items | length check | Top up with popularity fallback |
| Bad model after retrain | evaluation gate | Do not promote; keep previous version; log reason |
| Data leakage | split/feature tests | Unit test asserts max(train ts) ≤ min(val ts) and features use only train |
| Training memory blowups | OOM | Dev subset, batch size config, Spark partition tuning, Colab for full run |
| Version mismatch between index and ranker | version check | Artifacts bundled and loaded by one `model_version` |
