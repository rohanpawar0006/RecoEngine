# PRD — Large-Scale Personalized Recommendation Engine

**Version:** 1.0 · **Created:** 2026-11-04 · **Owner:** Rohan · **Context:** Nexvia AI/ML internship project (deadline **20 Nov 2026**, ~14 days effort)

**Legend:** `[USER]` = from the project brief · `[ASSUMPTION]` = my reasonable default, change if wrong · `[REC]` = recommendation · `[FUTURE]` = not now

---

## 1. Product Overview

- **Project name:** RecoEngine `[ASSUMPTION — working name]`
- **One-liner:** A two-stage recommender (fast candidate retrieval → precise deep ranking) of the kind used by Netflix/Amazon, with cold-start handling, caching, scheduled retraining, and an A/B comparison of two ranking approaches. `[USER]`
- **Vision:** Show that I can build and *evaluate honestly* a production-shaped ML system end to end, not just train a model in a notebook.
- **Problem:** Scoring every item for every user is too slow at scale, and a single model can't be both fast and precise. New users with no history also break pure collaborative filtering.
- **Proposed solution:**
  1. **Stage 1 – Retrieval:** two-tower model (user tower + item tower) → item embeddings indexed in **FAISS** → top few hundred candidates in milliseconds.
  2. **Stage 2 – Ranking:** a deep ranking model (DCN-V2 or Wide & Deep) re-scores the candidates with richer features.
  3. **Fallbacks:** popularity + content-based recommendations for cold-start users, cached in **Redis**.
  4. **Serving & ops:** model served through **Triton**, retraining scheduled in **Airflow**, features built with **Spark**, A/B comparison of retrieval-only vs retrieval+ranker.

## 2. Target Users

| Type | Who | Goals | Pain points |
|---|---|---|---|
| Primary | Nexvia evaluators (reporting manager Rishi, reviewers) | Judge technical depth, rigor, real results, working demo | Projects that only show a notebook, no evaluation, or fake metrics |
| Secondary | Recruiters / interviewers viewing GitHub + resume | Quickly see architecture, results, code quality | Unreadable repos, no README, unverifiable claims |
| Simulated end user | A viewer of movies (via demo UI) | Relevant top-10 list, including as a brand-new user | Irrelevant or generic recommendations, slow responses |
| Developer | Rohan (and future AI coding agent) | Clear docs, small phases, reproducible runs | Scope creep, environment issues, data leakage mistakes |

## 3. Goals & Objectives

- **Primary:** Deliver every item in the brief's checklist, to a standard that "goes beyond the minimum" `[USER]`: working demo, GitHub repo, architecture diagram + API docs, test results with real numbers, 10-minute presentation, individual reflection.
- **Secondary:** Strong, defensible offline metrics (NDCG@K, Recall@K, simulated click-through) with proper baselines; a clean story about cold start.
- **Technical objectives:**
  - Temporal (not random) train/validation/test split, with zero leakage.
  - Retrieval and ranking trained, evaluated and exported reproducibly.
  - End-to-end recommendation latency measured and reported.
  - Pipeline runs unattended via an Airflow DAG.
- **Product/business objective:** Earn the full-time offer signal the brief mentions by being rigorous and demonstrable `[USER]`.

## 4. Core Features

Dataset `[ASSUMPTION]`: **MovieLens 25M** (25M ratings, ~162k users, ~62k movies). It's the brief's first option, has timestamps (needed for time-based split), genres, and tags. Alternatives in the brief (Amazon Reviews 2023, Yelp, H&M, Goodbooks-10k, Last.fm) are out unless MovieLens proves unworkable.

Implicit-feedback definition `[ASSUMPTION]`: rating ≥ 4.0 = positive interaction (the "click/like" proxy).

### MVP

**F1. Data pipeline & temporal split**
- *Why:* Everything else depends on clean, leakage-free data. The brief explicitly demands a time-based split.
- *Behavior:* Load, clean, dedupe, map IDs to contiguous integers, split by timestamp into train / val / test.
- *Input → Output:* raw MovieLens CSVs → Parquet train/val/test + ID vocabularies + split report.
- *Depends on:* nothing.

**F2. Spark feature pipeline**
- *Why:* Brief requires Spark for interaction tables and features (user history, item popularity, categories).
- *Behavior:* Computes user features (interaction count, average rating, genre affinity, last-N item history), item features (popularity, average rating, genres, release year), all **only from the training window**.
- *Input → Output:* train Parquet → feature Parquet tables.
- *Depends on:* F1.

**F3. Two-tower retrieval model**
- *Why:* Fast first stage that scales to a huge catalog.
- *Behavior:* User tower and item tower (TensorFlow Recommenders) produce embeddings in a shared space; trained with in-batch negatives; item embeddings exported.
- *Input → Output:* interactions + features → trained towers + item embedding matrix.
- *Depends on:* F1, F2.

**F4. FAISS nearest-neighbor index**
- *Why:* Turns embeddings into millisecond candidate lookup.
- *Behavior:* Builds an index over item embeddings; given a user embedding returns top-N (N=200 `[ASSUMPTION]`) item IDs with scores. Exact index first; approximate (IVF/HNSW) only if worth comparing.
- *Input → Output:* item embeddings → index file; user embedding → candidate list.
- *Depends on:* F3.

**F5. Deep ranking model**
- *Why:* Re-scores candidates precisely using cross features that the two-tower can't use.
- *Behavior:* DCN-V2 (or Wide & Deep) predicts probability of positive interaction for (user, candidate); trained with hard negatives drawn from retrieval output.
- *Input → Output:* (user features, item features, context) → score; candidates → re-ranked list.
- *Depends on:* F2, F3, F4.

**F6. Cold-start fallbacks**
- *Why:* Brief requires it; new users have no embedding signal.
- *Behavior:* Unknown user → global popularity (time-decayed) list; if the user supplies a few liked genres/items → content-based (genre/tag similarity) list.
- *Input → Output:* user ID (unknown) and optional seed preferences → top-K list tagged `source=fallback`.
- *Depends on:* F1, F2.

**F7. Redis caching**
- *Why:* Brief requires caching; cuts latency and load.
- *Behavior:* Caches final recommendation lists per (variant, user), popularity lists, and hot user features with TTLs; cache misses fall through to the pipeline; Redis down = system still works (slower).
- *Input → Output:* user/variant key → cached list or miss.
- *Depends on:* F6 (and F8 for integration).

**F8. Model serving (Triton) + API (FastAPI)**
- *Why:* Brief requires Triton; an API makes the system demonstrable and testable.
- *Behavior:* Ranker (and user tower, if feasible) served from a Triton model repository; a FastAPI service orchestrates cache → retrieval → ranking → fallback and returns JSON.
- *Input → Output:* HTTP request `{user_id, k, variant}` → ranked items with scores, source, latency.
- *Depends on:* F3–F7.

**F9. Airflow retraining DAG**
- *Why:* Brief requires scheduled retraining.
- *Behavior:* DAG chains: build features → train retrieval → rebuild index → train ranker → evaluate → promote model only if metrics don't regress → refresh Triton model repo → flush cache. Demonstrated with at least one full successful run.
- *Input → Output:* schedule/trigger → versioned model artifacts + evaluation report.
- *Depends on:* F2–F5.

**F10. A/B comparison of two ranking approaches**
- *Why:* Brief requires a "simple online A/B test comparing two ranking models."
- *Behavior:* Deterministic user bucketing (hash of user ID) into **A = retrieval-score ordering** and **B = retrieval + deep ranker**. Compare NDCG, Recall@K and click-through. **Clicks are simulated** from held-out test interactions (replay) because there is no live traffic. This is documented honestly in results.
- *Input → Output:* test users → per-variant metrics + significance estimate (bootstrap confidence intervals).
- *Depends on:* F8.

**F11. Evaluation & reporting**
- *Why:* The brief: "test it well, and show real results."
- *Behavior:* One reproducible script produces all metric tables: popularity baseline, retrieval-only, retrieval+ranker, cold-start vs warm users; latency percentiles.
- *Input → Output:* test split + models → `reports/` tables, plots, JSON.
- *Depends on:* F3–F8.

**F12. Demo UI**
- *Why:* Deliverable is a working demo (deployed or screen-recorded) `[USER]`.
- *Behavior:* Streamlit app: pick a user (or "new user"), choose variant A/B, see recommendations, source tags, latency, cache hit/miss, and a metrics tab.
- *Input → Output:* UI selections → API calls → rendered cards/charts.
- *Depends on:* F8, F10, F11.

### Post-MVP `[REC]` (only if MVP is complete and verified)
- Per-segment metrics (heavy vs light users, genre slices); diversity/novelty metrics.
- Compare exact FAISS vs IVF/HNSW (recall vs latency curve).
- Second dataset to show generality (e.g., Goodbooks-10k).
- Model registry-style versioning, richer dashboards.

### Future possibilities `[FUTURE]`
Real-time feature updates, a feature store, GPU serving, true online experiments, multi-objective ranking, re-ranking for fairness/diversity.

## 5. User Stories

- As an **evaluator**, I want to open a demo and request recommendations for different users, so that I can see the system actually works.
- As an **evaluator**, I want a metrics report against baselines, so that I can trust the claimed improvement.
- As a **viewer**, I want relevant recommendations in under a second, so that the experience feels instant.
- As a **brand-new viewer**, I want sensible recommendations before I've rated anything, so that I'm not shown an empty or random list.
- As a **developer**, I want models to retrain on a schedule and only be promoted if they're not worse, so that quality doesn't silently regress.
- As a **developer**, I want a deterministic A/B split, so that variant comparison is reproducible.
- As a **recruiter**, I want a clear README and architecture diagram, so that I can understand the project in minutes.
- As a **developer**, I want phased docs and a Memory file, so that any AI agent can resume work without losing context.

## 6. Functional Requirements

- FR1. Split data by **time**, never randomly; no feature may use information from after the split point.
- FR2. Produce user and item embeddings in the same vector space; index item embeddings in FAISS.
- FR3. Retrieve top-N candidates (N configurable, default 200) for a known user.
- FR4. Re-rank candidates with the deep ranking model and return top-K (default 10).
- FR5. For unknown users, return fallback recommendations (popularity; content-based if seed preferences given), labeled by source.
- FR6. Cache results in Redis with TTLs; API must work when Redis is unavailable.
- FR7. Serve the ranker via Triton; API falls back to in-process inference if Triton is unavailable (flagged in response and logs).
- FR8. Expose a REST API: recommend, health, event logging, A/B summary.
- FR9. Assign variants by deterministic hash; log every impression/click event.
- FR10. Retraining DAG runs end-to-end and gates promotion on evaluation.
- FR11. Compute NDCG@K and Recall@K (K ∈ {10, 50, 200 for retrieval}) and simulated CTR, with baselines.
- FR12. Provide a demo UI exposing all of the above.

## 7. Non-Functional Requirements

| Area | Requirement |
|---|---|
| Performance | Targets `[ASSUMPTION]`: p95 ≤ 150 ms for cache-miss, warm-user recommendations on dev machine; ≤ 20 ms on cache hit. Measure and report actual numbers; adjust targets after Phase 7 measurement. |
| Scalability | Design must explain how it scales (ANN index, batching, caching, Spark) even though only MovieLens-scale is run. |
| Reliability | Graceful degradation: Redis down → no cache; Triton down → in-process; user unknown → fallback. |
| Security | No secrets in repo; input validation on API; see Architecture.md. |
| Accessibility | Demo UI readable, keyboard-usable, sufficient contrast (see Design.md). |
| Maintainability | Modular `src/` package, typed, tested, config-driven. |
| Privacy | Public dataset only; no personal data. Treat Nexvia-specific material as confidential. |
| Observability | Structured logs; latency, cache-hit rate, fallback rate, variant counts. |
| Reproducibility | Fixed seeds, pinned dependencies, one command per pipeline stage. |

## 8. Success Criteria

Measured baselines on validation split (Phase 1, evaluated on 10,000 validation users with positive interactions):
- **Random Baseline:** Recall@200 = 0.00532, NDCG@10 = 0.00208
- **Popularity Baseline:** Recall@200 = 0.33366, NDCG@10 = 0.29482

Target criteria for subsequent stages:
1. Retrieval Recall@200 **beats** the popularity baseline (Recall@200 > 0.33366) by a documented margin.
2. Retrieval+ranker NDCG@10 **beats** retrieval-only ordering, with bootstrap confidence intervals reported.
3. Cold-start users get fallback recommendations that **beat random** (Recall@200 > 0.00532, NDCG@10 > 0.00208) and are compared against popularity on the cold-start slice of the test set.
4. End-to-end latency percentiles (p50/p95/p99) reported for cache hit and miss.
5. Airflow DAG completes one full successful run, with evidence (screenshot/log).
6. All deliverables in the brief's checklist exist and are in the shared Google Drive folder.
7. A fresh clone can reproduce the main results from the README.

## 9. Constraints

- **Time:** hard deadline 20 Nov 2026; ~14 working days; plan includes a 2-day buffer.
- **Budget:** zero; free tools and free GPU tiers (Colab/Kaggle) only `[ASSUMPTION]`.
- **Hardware `[ASSUMPTION]`:** personal laptop, no dedicated GPU; heavy training on Colab/Kaggle, development on a data subset.
- **Technology (from brief):** TensorFlow Recommenders, FAISS, Redis, Apache Spark, Airflow, Triton Inference Server.
- **Platform `[ASSUMPTION]`:** Windows laptop. Triton and Airflow are Linux-first → run via Docker/WSL2.
- **Python:** profile shows Python 3.13; TensorFlow/TFRS compatibility with 3.13 must be verified in Phase 0 — otherwise use a separate 3.11/3.12 environment.
- **Confidentiality:** offer letter says work product belongs to Nexvia and company information is confidential → confirm with the manager before publishing the repo publicly (see Open Questions).

## 10. Out of Scope

- Real production deployment, real user traffic, true online A/B testing.
- Training on proprietary Nexvia data (none provided).
- Multi-dataset benchmark suite, auto-hyperparameter-search platforms, feature stores.
- Mobile apps, authentication/user accounts in the demo, payments.
- LLM-based recommenders.
- Movie posters or external metadata APIs.

## 11. Open Questions

1. May the repo be public / used in the portfolio, given the offer letter's IP and confidentiality clause? *(ask Rishi)*
2. Does Nexvia prefer a specific dataset among the six options? (Default: MovieLens 25M.)
3. "Deployed or screen-recorded" demo — which is preferred? (Default: screen recording + local Docker Compose.)
4. Is the Google Drive folder structure prescribed?
