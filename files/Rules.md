# Rules — RecoEngine Development & AI Coding Rules

**Version:** 1.0 · **Created:** 2026-11-04 · Binding for any developer or AI coding agent unless Rohan explicitly says otherwise.

---

## 1. General Coding Rules

- **Quality:** small functions, one responsibility each, no dead code, no commented-out blocks.
- **Naming:** `snake_case` for functions/variables/modules, `PascalCase` for classes, `UPPER_CASE` constants. Names state intent (`build_item_index`, not `process`).
- **File organization:** follow `Architecture.md` folder structure exactly. New top-level folders require a documented reason.
- **Config over constants:** paths, hyperparameters, `k`, TTLs, seeds live in `configs/*.yaml`, not scattered in code.
- **Documentation:** every public function has a docstring (purpose, args, returns). Every module has a one-line header. Non-obvious ML decisions (e.g., why rating ≥ 4) get a comment pointing to `Memory.md` decision log.
- **Type safety:** type hints on all public functions; Pydantic models for API I/O.
- **Logging:** use `logging` (never bare `print` outside scripts/notebooks); structured, with model version and request IDs in serving.
- **Reproducibility:** set seeds (Python, NumPy, TF) from config; record config + git commit hash with every training run's output.
- **Testing:** every phase ships with tests (see Phases.md). A task is not done until its tests pass.
- **Notebooks:** EDA/visualization only. Any logic the pipeline depends on lives in `src/`.

## 2. Technology Rules

### MUST USE (from the brief)
TensorFlow Recommenders (retrieval + ranking), FAISS, Redis, Apache Spark, Airflow, Triton Inference Server.

### MUST USE (project choices)
Python with a **verified-compatible** TF/Keras/TFRS combination (pinned in `requirements.txt`), Parquet for tables, FastAPI + Pydantic for the API, pytest for tests, Docker Compose for services, a **temporal** split.

### PREFER
- `faiss-cpu`, exact index first.
- Streamlit for the demo.
- ruff for lint/format.
- Dev subset (e.g., ~1M interactions) for iteration; full dataset only for final runs.
- Simple models first (baseline → two-tower → ranker), measured at each step.

### DO NOT USE
- **Random train/test split** for the main evaluation.
- Any feature/statistic computed using val/test data (leakage).
- Test set for tuning. Test set is evaluated for reporting only, after val-based decisions are frozen.
- `pickle` for sharing artifacts; use NPY/JSON/SavedModel/FAISS formats.
- Unrequested heavy frameworks (no PyTorch, LangChain, Kubernetes, MLflow in MVP, etc.).
- Hardcoded absolute paths or credentials.

**New dependency rule:** never add a dependency without stating (a) why it's needed, (b) what was considered instead, (c) updating `requirements.txt` and `Memory.md`.

## 3. Architecture Rules

The AI must:
- Follow `Architecture.md`; if something doesn't fit, propose an update first (see Change Management).
- Keep stages separated: data → features → retrieval → index → ranking → serving → evaluation. No stage imports from a later stage.
- Reuse existing utilities (`utils/`, metric functions); no duplicate logic.
- Avoid abstractions with only one use.
- Treat model + index + ranker as **one versioned bundle** (`artifacts/<version>/`).
- Keep serving stateless; state lives in Redis/files.

## 4. AI Coding Rules

When modifying the project:
1. Understand existing code before changing it.
2. Read `Memory.md`, `Phases.md`, and the relevant parts of `PRD.md`/`Architecture.md` before starting.
3. Confirm the task belongs to the **current phase**.
4. Follow the current architecture.
5. Don't rewrite working code unnecessarily.
6. Don't add dependencies without justification.
7. Don't change API contracts without updating `Architecture.md` and tests.
8. Don't silently change requirements.
9. Don't remove existing functionality unless told to.
10. Keep changes focused; implement the smallest appropriate change.
11. Run tests/commands before declaring done, and report real output.
12. Update `Memory.md` after every meaningful task.

## 5. Error Handling

- **Detect:** validate inputs at boundaries (API, data loading); check shapes/dtypes/NaNs after each data stage.
- **Handle:** degrade gracefully where the architecture defines a fallback (Redis, Triton, cold start); otherwise fail fast with a clear message.
- **Log:** every caught exception logged with context; fallbacks logged at WARNING.
- **Display:** users see short, safe messages; developers see details in logs.
- **Retry:** only idempotent network calls (Redis/Triton), max 2 retries with short backoff; never retry silently forever.
- **Escalate:** Airflow task failure → DAG stops, nothing is promoted, failure recorded in `Memory.md` Known Issues.
- **Never hide errors silently** (no bare `except:`; no `except Exception: pass`).

## 6. Security Rules

Never: hardcode secrets, commit `.env`/keys, expose Redis/Triton publicly, trust unvalidated input, bypass auth if added, disable a security control to make something work, commit raw datasets or Nexvia-confidential material.

## 7. ML / Evaluation Rules

(Replaces the "AI/LLM rules" in the template — this is a classical ML system, no LLM.)

- **Baselines are mandatory:** random, global popularity, retrieval-only. Every claimed improvement is relative to a baseline.
- **Metrics defined once** in `evaluation/metrics.py` and tested against a tiny hand-computed example.
- **Split integrity test** must exist and pass: timestamps ordered; no user/feature leakage.
- **Report honestly:** show confidence intervals (bootstrap) for key comparisons; report negative or null results.
- **A/B simulation must be labeled as simulation** everywhere (README, report, presentation, UI). Never describe it as a real online experiment.
- **No fabricated numbers.** Metrics in docs/README/slides must come from a file in `reports/` produced by the evaluation script.
- **Model gating:** a retrained model is promoted only if validation NDCG@10 is not worse than the current model beyond a configured tolerance.
- **Cold-start metrics reported separately** from warm-user metrics.
- **Token/prompt rules:** N/A.
- **The AI must not claim an action was completed if it was not actually completed** (training run, test pass, DAG run, benchmark). Say what was run, what was observed, and what wasn't verified.

## 8. Git & Delivery Rules

- Small commits with meaningful messages (`feat(retrieval): add two-tower training loop`).
- One branch per phase is optional; `main` must always run.
- README must stay accurate: setup, one-command-per-stage, results table linking to `reports/`.
- Don't push data, models, or secrets. Use `.gitignore`.
- Public visibility of the repo depends on the Nexvia confidentiality answer (PRD Open Question 1).

## 9. Change Management

Before any major change, state:
1. What is changing
2. Why
3. Files affected
4. Risks
5. Whether docs (PRD → Architecture → Rules → Phases → Design → Memory) need updating

Then update the docs **in that order** so they never contradict each other. If two docs conflict, flag it and recommend which to update — don't silently pick one.
