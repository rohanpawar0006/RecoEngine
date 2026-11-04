# Memory — RecoEngine Project State

**Last updated:** 2026-11-04 · **Update this file after every meaningful task.** Never store secrets here.

---

## Project Status

- **Current phase:** Phase 0 — Setup & Environment (not started)
- **Current milestone:** Docs foundation complete
- **Overall completion:** ~0% implementation (docs 100%)
- **Last completed task:** Generated PRD, Architecture, Rules, Phases, Design, Memory
- **Current task:** Verify environment (Python vs TensorFlow/TFRS compatibility, Docker/WSL2, Java for Spark)
- **Next task:** Create repo skeleton, pin dependencies, download MovieLens 25M
- **Deadline:** 20 Nov 2026 (buffer days 19–20)

## Completed Work

```text
[✓] Project brief analyzed
[✓] PRD.md, Architecture.md, Rules.md, Phases.md, Design.md, Memory.md written
[ ] Phase 0  Setup & environment
[ ] Phase 1  Data, temporal split, baselines
[ ] Phase 2  Spark features
[ ] Phase 3  Two-tower retrieval
[ ] Phase 4  FAISS index + retrieval eval
[ ] Phase 5  Deep ranking model
[ ] Phase 6  Cold-start + Redis
[ ] Phase 7  FastAPI + Triton serving
[ ] Phase 8  Airflow retraining DAG
[ ] Phase 9  A/B simulation + final evaluation
[ ] Phase 10 Demo, docs, presentation, reflection
```

## Current Architecture (summary)

Two-stage recommender on MovieLens 25M: two-tower retrieval (TFRS) → FAISS top-200 → deep ranker (DCN-V2 or Wide & Deep) → top-K. Cold-start fallback (popularity/content-based). Redis cache. Triton serves models; FastAPI orchestrates. Spark builds features; Airflow retrains with a promotion gate. Streamlit demo. Details in `Architecture.md`.

## Important Decisions

| Decision | Reason | Alternatives rejected |
|---|---|---|
| MovieLens 25M as dataset (assumption) | Brief's first option; has timestamps, genres, tags | Amazon Reviews 2023, Yelp, H&M, Goodbooks-10k, Last.fm (heavier or fewer features) |
| Positive = rating ≥ 4 (assumption) | Standard implicit-feedback proxy | Using all ratings as positives |
| Temporal split ~80/10/10 | Required by brief; avoids leakage; yields natural cold-start users | Random split |
| Candidate size N = 200 (assumption) | Brief says "a few hundred" | 100, 500 |
| A/B is a **replay simulation**, labeled as such | No live traffic available | Claiming a real online test |
| Streamlit for demo | Fast to build, good enough for evaluators | React frontend |
| Docker Compose for Redis/Triton/Airflow | Linux-first tools, Windows laptop (assumption) | Native installs |
| No relational DB | Files + Redis suffice | PostgreSQL |

## Known Issues / Risks

| Issue | Severity | Status | Possible solution |
|---|---|---|---|
| Python 3.13 may not be supported by TF/TFRS | High | Unverified | Separate 3.11/3.12 venv |
| Triton/Airflow on Windows | High | Unverified | Docker/WSL2; cut-ladder step 4 if blocked |
| No GPU for full training | Medium | Open | Dev subset locally; Colab/Kaggle for full |
| Spark needs Java and memory tuning | Medium | Unverified | Install JDK, cap driver memory, tune partitions |
| Offer letter says work belongs to Nexvia + confidentiality | Medium | **Open question for Rishi** | Confirm before making repo public / using in portfolio |
| Timeline is tight (14 days) | High | Open | Follow Scope-Cut Ladder in Phases.md |

## Technical Debt

None yet.

## Environment

- **Machine:** personal laptop, assumed Windows, no GPU `[ASSUMPTION — confirm]`
- **IDE:** VS Code; Python version on machine 3.13 (compat TBD)
- **Services (planned, Docker):** Redis `6379`, Triton `8000/8001/8002`, Airflow webserver `8080`, FastAPI `8100` (avoids Triton/Airflow clashes), Streamlit `8501`
- **Env vars (names only):** `REDIS_URL`, `TRITON_URL`, `MODEL_VERSION`, `DATA_DIR`, `ARTIFACT_DIR`, `API_KEY` (only if exposed publicly)
- **Commands:** TBD in Phase 0 (one script per stage under `scripts/`)
- **Pinned versions:** TBD in Phase 0

## Results Log (fill only from `reports/`)

| Metric | Random | Popularity | Retrieval | Retrieval+Ranker |
|---|---|---|---|---|
| Recall@200 | TBD | TBD | TBD | n/a |
| NDCG@10 | TBD | TBD | TBD | TBD |
| p95 latency (miss/hit) | – | – | – | TBD |

## Recent Changes

- 2026-11-04 — Documentation system created.

## Next Steps

1. Phase 0 tasks (see `Phases.md`).
2. Send Rishi the open questions (public repo, dataset, demo format).
3. Download MovieLens 25M and start Phase 1.

---

## Context for the Next AI Session

- **Project:** RecoEngine — two-stage recommender (two-tower retrieval + FAISS + deep ranker) for Rohan's Nexvia AI/ML internship; **deadline 20 Nov 2026**.
- **Source of truth:** the six docs in `docs/`; priority order: explicit user instruction → PRD → Architecture → Rules → Phases → Design → Memory.
- **State:** documentation complete; **no code written**; Phase 0 is next.
- **Hard rules:** temporal split only; no leakage; baselines before models; every reported number comes from `reports/`; A/B is a labeled simulation; never claim something ran if it didn't; no secrets in repo.
- **Biggest risks:** TF/TFRS vs Python 3.13, Triton/Airflow on Windows (use Docker/WSL2), tight timeline (use the Scope-Cut Ladder), repo confidentiality question.
- **Rohan's preferences:** structured, actionable material; program-by-program style with explanations and viva-style prep helps him; keep steps small and testable.
- **Update this section after every meaningful task.**
