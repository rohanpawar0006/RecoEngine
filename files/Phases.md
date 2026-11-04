# Phases — RecoEngine Implementation Roadmap

**Version:** 1.0 · **Created:** 2026-11-04 · **Deadline:** 20 Nov 2026

**Schedule `[ASSUMPTION]`:** Day 1 = 5 Nov → Day 14 = 18 Nov; **19–20 Nov = buffer** (final checks, upload to Drive). If a phase slips, use the *Scope-Cut Ladder* at the bottom instead of skipping evaluation.

Rule: do not start Phase N+1 until Phase N meets its acceptance criteria (unless Rohan says otherwise).

| Phase | Name | Target day |
|---|---|---|
| 0 | Setup & environment | Day 1 (5 Nov) |
| 1 | Data, temporal split, baselines | Day 2 (6 Nov) |
| 2 | Spark feature pipeline | Day 3 (7 Nov) |
| 3 | Two-tower retrieval model | Days 4–5 (8–9 Nov) |
| 4 | FAISS index + retrieval evaluation | Day 6 (10 Nov) |
| 5 | Deep ranking model | Days 7–8 (11–12 Nov) |
| 6 | Cold-start fallbacks + Redis | Day 9 (13 Nov) |
| 7 | Serving: FastAPI + Triton | Days 10–11 (14–15 Nov) |
| 8 | Airflow retraining DAG | Day 12 (16 Nov) |
| 9 | A/B simulation + final evaluation | Day 13 (17 Nov) |
| 10 | Demo, docs, presentation, reflection | Day 14 (18 Nov) |

---

## Phase 0 — Setup & Environment

**Objective:** A reproducible repo and a verified environment (the biggest early risk).
**Features:** repo skeleton, pinned dependencies, Docker services, docs committed.
**Files/Components:** whole folder tree from `Architecture.md`, `requirements.txt`, `docker-compose.yml`, `.gitignore`, `.env.example`, `docs/`.
**Dependencies:** none.

**Tasks**
- [ ] Create repo `reco-engine` with the folder structure; commit the six docs into `docs/`
- [ ] Check Python version vs TensorFlow + TFRS compatibility; create a 3.11/3.12 venv if 3.13 is unsupported
- [ ] Install and import-test: tensorflow, tensorflow-recommenders, faiss-cpu, pyspark (Java present), redis, fastapi, streamlit, pytest
- [ ] Docker Desktop/WSL2 working; `docker compose up` starts Redis and returns `PONG`
- [ ] Decide Triton + Airflow approach (Docker); pull images
- [ ] Download MovieLens 25M into `data/raw/`
- [ ] Set up ruff, pytest, `.gitignore` (data/, artifacts/, .env)
- [ ] Ask Rishi the open questions (public repo? dataset? demo format?)

**Testing:** smoke test script imports every library and pings Redis.
**Acceptance:** `pytest` runs (even with one trivial test); all libs import; Redis reachable; versions recorded.
**Docs to update:** Memory.md (environment, versions), Rules.md (final pinned versions), Architecture.md if any stack item changes.

---

## Phase 1 — Data, Temporal Split, Baselines

**Objective:** Clean data, leakage-free splits, and baseline numbers everything else must beat.
**Features:** F1, F11 (skeleton), baselines.
**Files:** `src/reco/data/*`, `src/reco/evaluation/metrics.py`, `scripts/prepare_data.py`, `tests/test_split.py`, `tests/test_metrics.py`.
**Dependencies:** Phase 0.

**Tasks**
- [ ] Load ratings/movies/tags; clean, dedupe; define positives (rating ≥ 4)
- [ ] Build user/item ID vocabularies (train-only fit; unseen = OOV/cold)
- [ ] Temporal split (~80/10/10 by timestamp); save Parquet
- [ ] Create a **dev subset** (~1M interactions) with the same split logic
- [ ] Implement NDCG@K and Recall@K; unit-test on a hand-computed example
- [ ] Implement baselines: random, global popularity (train-only)
- [ ] Report split stats: sizes, date ranges, % cold-start users/items in val/test

**Testing:** split ordering test (max train ts ≤ min val ts ≤ min test ts); metric tests; vocab built only from train.
**Acceptance:** split report generated; baselines' NDCG@10 / Recall@K computed on val and saved to `reports/baselines.json`.
**Docs:** Memory.md (baseline numbers, decisions on positive threshold & split ratios); PRD success-criteria numbers filled from baselines.

---

## Phase 2 — Spark Feature Pipeline

**Objective:** Interaction tables and features computed with Spark, training-window only.
**Features:** F2.
**Files:** `src/reco/features/*`, `scripts/build_features.py`, `tests/test_features.py`.
**Dependencies:** Phase 1; Java + PySpark working.

**Tasks**
- [ ] Spark session config (local[*], memory limit from config)
- [ ] User features: interaction count, avg rating, genre affinity, last-N item history
- [ ] Item features: popularity (train window), avg rating, genres, release year
- [ ] Write feature Parquet tables; document schema
- [ ] Verify no feature uses val/test rows

**Testing:** leakage test (recompute on a tiny table and compare); null/NaN checks; row-count checks.
**Acceptance:** feature tables exist for full data and dev subset; leakage test passes; schema documented.
**Docs:** Architecture.md (feature schema), Memory.md.

---

## Phase 3 — Two-Tower Retrieval Model

**Objective:** Trained retrieval model that beats popularity on Recall@K.
**Features:** F3.
**Files:** `src/reco/retrieval/*`, `scripts/train_retrieval.py`, `tests/test_retrieval_shapes.py`, `configs/retrieval.yaml`.
**Dependencies:** Phases 1–2.

**Tasks**
- [ ] User tower (ID embedding + history/features), item tower (ID embedding + genres/features)
- [ ] TFRS retrieval task with in-batch negatives
- [ ] Train on dev subset first, then full data (Colab/Kaggle if needed)
- [ ] Evaluate val Recall@10/50/200 vs baselines
- [ ] Save towers and item embedding matrix to `artifacts/<version>/`
- [ ] Record hyperparameters + seed + commit hash

**Testing:** embedding dimension test; deterministic output for fixed seed; embeddings finite.
**Acceptance:** model trains end-to-end; Recall@200 on val **>** popularity baseline; artifacts saved and reloadable.
**Docs:** Memory.md (results, decisions, rejected options), Architecture.md if tower design changed.

---

## Phase 4 — FAISS Index + Retrieval Evaluation

**Objective:** Fast, verified nearest-neighbor retrieval.
**Features:** F4.
**Files:** `src/reco/indexing/*`, `scripts/build_index.py`, `tests/test_index.py`.
**Dependencies:** Phase 3.

**Tasks**
- [ ] Build exact (inner-product) FAISS index over item embeddings; save/load
- [ ] Search wrapper (user embedding → top-200 IDs + scores); filter already-seen items
- [ ] Verify FAISS results equal brute-force numpy on a sample
- [ ] Benchmark search latency (batch and single)
- [ ] *(optional)* try IVF/HNSW and record recall-vs-latency

**Testing:** FAISS vs brute-force equality; save/load roundtrip.
**Acceptance:** retrieval returns 200 candidates per user; latency measured and saved to `reports/`; test passes.
**Docs:** Memory.md, Architecture.md (index type chosen).

---

## Phase 5 — Deep Ranking Model

**Objective:** Ranker that beats retrieval-only ordering on NDCG@10.
**Features:** F5.
**Files:** `src/reco/ranking/*`, `scripts/train_ranker.py`, `tests/test_ranker.py`, `configs/ranking.yaml`.
**Dependencies:** Phases 2–4.

**Tasks**
- [ ] Build ranking dataset: positives + hard negatives from retrieval candidates
- [ ] Implement DCN-V2 (or Wide & Deep); optionally compare both
- [ ] Train with val-based early stopping; evaluate NDCG@10 re-ranking of retrieved candidates
- [ ] Compare: retrieval-only vs retrieval+ranker (val), then test once
- [ ] Export SavedModel for Triton

**Testing:** input/output shape test; score monotonic sanity check; exported model reload test.
**Acceptance:** ranker NDCG@10 on val **>** retrieval-only ordering; model exported and reloadable.
**Docs:** Memory.md, Architecture.md (feature list, model choice).

---

## Phase 6 — Cold-Start Fallbacks + Redis

**Objective:** Every user gets a sensible answer; repeated requests are fast.
**Features:** F6, F7.
**Files:** `src/reco/fallback/*`, `src/reco/serving/cache.py`, `tests/test_fallback.py`, `tests/test_cache.py`.
**Dependencies:** Phases 1–2, Redis running.

**Tasks**
- [ ] Time-decayed global popularity list; per-genre popularity
- [ ] Content-based recommender (genres/tags similarity) from seed likes
- [ ] Redis cache layer with TTL + versioned keys + graceful failure
- [ ] Evaluate fallback on the cold-start slice of val vs popularity vs random

**Testing:** unknown user → fallback; Redis down → no crash; TTL set; versioned key test.
**Acceptance:** cold-start evaluation table exists; cache hit/miss verified; Redis-down test passes.
**Docs:** Memory.md, Architecture.md (key schema).

---

## Phase 7 — Serving: FastAPI + Triton

**Objective:** One API serving the full pipeline, with the ranker (and user tower if feasible) in Triton.
**Features:** F8.
**Files:** `src/reco/serving/*`, `triton/model_repository/*`, `docker-compose.yml`, `tests/test_api.py`.
**Dependencies:** Phases 4–6.

**Tasks**
- [ ] Triton model repository + `config.pbtxt` for the ranker; confirm via `tritonclient`
- [ ] FastAPI: `/health`, `/recommend`, `/events`, `/ab/summary` with Pydantic schemas
- [ ] Orchestration: cache → user tower → FAISS → ranker → filter → fallback
- [ ] In-process fallback if Triton is down (flagged in response)
- [ ] Latency benchmark: p50/p95/p99, cache hit vs miss, warm vs cold users
- [ ] Export OpenAPI spec for API documentation deliverable

**Testing:** API tests (valid/invalid inputs); Triton-down and Redis-down tests; response-schema tests; ranking via Triton equals in-process scores within tolerance.
**Acceptance:** `docker compose up` brings the stack up; curl/`/docs` returns recommendations; latency report saved.
**Docs:** Architecture.md (final API), Memory.md.

> Risk: Triton on Windows. Use Docker/WSL2. If Triton can't be made to work in ~1 day, trigger the Scope-Cut Ladder (serve ranker in-process, document Triton config and attempt honestly).

---

## Phase 8 — Airflow Retraining DAG

**Objective:** Automated, gated retraining.
**Features:** F9.
**Files:** `airflow/dags/retrain_dag.py`, compose services for Airflow, `tests/test_dag_import.py`.
**Dependencies:** Phases 2–5, 7.

**Tasks**
- [ ] DAG: build_features → train_retrieval → build_index → train_ranker → evaluate → gate → export_to_triton → flush_cache
- [ ] Gate: promote only if val NDCG@10 ≥ current − tolerance
- [ ] Run on dev subset for speed in the demo; document full-data run
- [ ] Capture evidence of one successful run (graph view screenshot + logs) and one gated rejection (simulated)

**Testing:** DAG imports with no errors; task dependency order test; gate logic unit test.
**Acceptance:** one complete successful DAG run; gate behaviour demonstrated.
**Docs:** Architecture.md, Memory.md.

---

## Phase 9 — A/B Simulation + Final Evaluation

**Objective:** The brief's A/B comparison and all final numbers, honestly labeled.
**Features:** F10, F11.
**Files:** `src/reco/serving/ab.py`, `src/reco/evaluation/ab_replay.py`, `scripts/run_final_eval.py`, `reports/*`.
**Dependencies:** Phase 7.

**Tasks**
- [ ] Deterministic hash bucketing (A = retrieval-only, B = retrieval+ranker)
- [ ] Replay held-out test interactions as simulated clicks; log events through the API path
- [ ] Final test-set evaluation (once): baselines, retrieval, ranker, cold vs warm
- [ ] Bootstrap confidence intervals for A vs B
- [ ] Generate tables and plots for the report/presentation

**Testing:** bucketing determinism and ~50/50 balance test; replay accounting test.
**Acceptance:** `reports/final_results.md|json` exists, reproducible from one command; every number in README/slides traces to it.
**Docs:** PRD success criteria marked pass/fail; Memory.md.

---

## Phase 10 — Demo, Documentation, Presentation, Reflection

**Objective:** Package every deliverable in the brief.
**Features:** F12 + deliverables checklist.
**Files:** `demo/app.py`, `README.md`, `docs/architecture.png`, `docs/api.md`, slides, reflection.
**Dependencies:** Phase 9.

**Tasks**
- [ ] Streamlit demo per `Design.md` (user picker, new-user mode, A/B toggle, metrics tab)
- [ ] Record screen demo (2–3 min) and/or deploy
- [ ] README: overview, architecture diagram, setup, per-stage commands, results table, limitations
- [ ] Architecture diagram + API docs exported
- [ ] 10-minute presentation: problem → design → challenges → results → next steps
- [ ] Individual reflection (what I learned, what I owned)
- [ ] Fresh-clone test: follow README on a clean checkout
- [ ] Upload everything to the single Google Drive folder; resume project bullet updated *(only with real numbers)*

**Testing:** fresh-clone reproduction; demo smoke test.
**Acceptance:** every item in the brief's *Deliverables Checklist* is present and verified.
**Docs:** Memory.md final state; all docs consistent.

---

## Scope-Cut Ladder (use in this order if behind schedule)

1. Drop optional extras (IVF/HNSW comparison, second ranker, second dataset).
2. Serve only the ranker via Triton (user tower in-process).
3. Run Airflow DAG on dev subset only.
4. Serve ranker in-process; keep Triton config + honest write-up of what was attempted.
5. Reduce demo UI to the minimum (recommend + variant toggle + metrics table).

**Never cut:** temporal split, baselines, honest evaluation, README, presentation, reflection.
