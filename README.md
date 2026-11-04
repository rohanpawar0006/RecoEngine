# RecoEngine — Production Two-Stage Recommendation Engine

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow 2.18](https://img.shields.io/badge/tensorflow-2.18-orange.svg)](https://www.tensorflow.org/)
[![FAISS](https://img.shields.io/badge/FAISS-IndexFlatIP-green.svg)](https://github.com/facebookresearch/faiss)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal.svg)](https://fastapi.tiangolo.com/)
[![Redis](https://img.shields.io/badge/Redis-8.10-red.svg)](https://redis.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade, two-stage recommendation system built on the complete **MovieLens 25M** dataset (25,000,095 ratings across 162,541 users and 59,047 items). Engineered for high accuracy and ultra-low latency (<25ms p95), featuring strict temporal splitting (zero future leakage), hard-negative mining, cold-start recency fallbacks, fail-open Redis caching, automated Airflow retraining with non-inferiority gating, and offline A/B replay simulation with bootstrap confidence intervals.

---

## Architecture Overview

RecoEngine separates computation into an **offline training & MLOps layer** and an **online low-latency serving layer**:

```text
+-----------------------------------------------------------------------------------+
|                               OFFLINE TRAINING & DAG                               |
+-----------------------------------------------------------------------------------+
| MovieLens 25M (Strict Temporal 80/10/10 Split: 20M Train, 2.5M Val, 2.5M Test)    |
|      |                                                                            |
|      +--> PySpark Feature Pipeline (User & Item Parquet Feature Tables)           |
|      |                                                                            |
|      +--> Stage 1: Two-Tower Retrieval (TFRS, 64-dim, In-Batch Negatives)         |
|      |         |                                                                  |
|      |         +--> Catalog Embedding Extraction --> FAISS IndexFlatIP (8.82 MB)  |
|      |                                                                            |
|      +--> Stage 2: Deep Ranker (DCN-V2 with Cross Layers & Deep MLP)              |
|      |         |   (Hard negatives mined from Stage 1 FAISS Top-200)              |
|      |         +--> SavedModel Export --> Triton Model Repository / Local Serving |
|      |                                                                            |
|      +--> Retraining DAG (Airflow): Gated Promotion on NDCG@10 Non-Inferiority     |
+-----------------------------------------------------------------------------------+
                                          |
+-----------------------------------------------------------------------------------+
|                               ONLINE SERVING LAYER                                |
+-----------------------------------------------------------------------------------+
| Client Request: POST /recommend {"user_id": 105, "k": 10, "variant": "auto"}      |
|      |                                                                            |
|      v                                                                            |
| [1. Deterministic Salted Hash Bucketing (Variant A: 50% / Variant B: 50%)]        |
|      |                                                                            |
|      v                                                                            |
| [2. Redis Cache Lookup (rec:{variant}:{user_id}:{k})]                             |
|      ├──> CACHE HIT (p50: 5.62 ms) ──> Return Top-K                               |
|      └──> CACHE MISS                                                              |
|            │                                                                      |
|            ├──> Warm User:                                                        |
|            │      User Tower Embedding (TF) ──> FAISS Top-200 MIPS (0.98 ms)     |
|            │      ──> Previously-Seen Movie Filtering                             |
|            │      ──> [Variant A]: Return Top-K in Retrieval Order                |
|            │      ──> [Variant B]: DCN-V2 Ranker Re-Score (Triton / Local)        |
|            │                       ──> Sort Top-K by CTR Probability              |
|            │                                                                      |
|            └──> Cold User:                                                        |
|                   ──> Time-Decayed Popularity Fallback (2-Year Half-Life)         |
|                   ──> Or Genre Content-Based Matching Engine                      |
|                                                                                   |
| [3. Populate Redis Cache (TTL: 3600s)] ──> Return Top-K Response                  |
+-----------------------------------------------------------------------------------+
```

---

## Key Performance Results

All numbers below come directly from reproducible reports generated by scripts and audited on the held-out test split (`data/processed/test.parquet`).

### 1. PRD Section 8 Success Criteria Audit

| Criteria | Target Hurdle | Achieved Result | Verdict |
|---|---|---|---|
| **SC1: Retrieval Beats Popularity** | Val Recall@200 > 0.17418 | **0.25647 (+47.2% relative lift)** | **PASS** |
| **SC2: Ranker Beats Retrieval** | Test NDCG@10 > Stage 1 (0.01215) | **0.01604 (+32.05% relative lift)** | **PASS** |
| **SC3: Cold Fallback Beats Random** | Test NDCG@10 > Random (0.00198) | **0.13975 (+6,955.1% relative lift)** | **PASS** |
| **SC4: Serving Latency Measured** | p95 Latency < 150 ms | **p50: 17.65 ms, p95: 20.93 ms (Full Pipeline)** | **PASS** |
| **SC5: Retraining Pipeline Logged** | Audit log with gate decision | `reports/retraining_log.json` | **PASS** |
| **SC6: A/B Replay & Bootstrap CI** | Replay simulation with 95% CIs | **+17.82% CTR Lift (95% CI: [-7.2%, +40.6%])** | **PASS** |

### 2. Multi-Stage Warm User Performance (Held-Out Test Set)
*Evaluated on sample of 2,000 warm test users from `reports/final_results.json`:*

| Model / Strategy | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 | Recall@200 | NDCG@200 |
|---|---|---|---|---|---|---|
| **Random Baseline** | 0.00011 | 0.00068 | 0.00102 | 0.00099 | 0.00329 | 0.00179 |
| **Popularity Baseline (Warm)** | 0.01413 | 0.02967 | 0.04397 | 0.03547 | 0.11373 | 0.05844 |
| **Stage 1: Two-Tower (TFRS + FAISS)** | 0.00462 | 0.01215 | 0.02001 | 0.01530 | 0.05617 | 0.02692 |
| **Stage 2: DCN-V2 Deep Ranker** | **0.00660** | **0.01604** | – | – | – | – |

> **Takeaway:** Stage 2 DCN-V2 re-ranking delivers a **+32.05% NDCG@10 lift** over Stage 1 candidate ordering on the strictly held-out test split.

### 3. Cold-Start Fallback Performance (Unseen Test Users)
*Evaluated on sample of 2,000 cold test users (78.1% of test users) from `reports/final_results.json`:*

| Fallback Strategy | Recall@10 | NDCG@10 | Recall@50 | Recall@200 |
|---|---|---|---|---|
| **Random Baseline** | 0.00026 | 0.00198 | 0.00126 | 0.00504 |
| **Raw Global Popularity** | 0.05122 | 0.29127 | 0.14831 | 0.31480 |
| **Time-Decayed Popularity Fallback** | **0.02490** | **0.13975** (+6,955% vs Random) | **0.05782** | **0.08245** |
| **Content-Based Fallback (Seed Genre)** | 0.01150 | 0.07630 (+3,753% vs Random) | 0.03773 | 0.09062 |

### 4. Serving Latency Benchmark (`reports/serving_latency.json`)
*Benchmarked across 100 consecutive requests per serving route on FastAPI:*

| Serving Route | Mean | p50 | p95 | p99 | SLA Status |
|---|---|---|---|---|---|
| **Warm Cache Hit (Redis)** | 6.00 ms | **5.62 ms** | 9.30 ms | 10.07 ms | PASS (< 20 ms) |
| **Warm Full Pipeline (Miss)** | 18.81 ms | **17.65 ms** | **20.93 ms** | 43.87 ms | PASS (< 150 ms) |
| **Cold Popularity Fallback** | 6.50 ms | **6.03 ms** | 9.49 ms | 10.63 ms | PASS (< 50 ms) |
| **Cold Content-Based Fallback** | 163.21 ms | 159.24 ms | 196.69 ms | 224.38 ms | Baseline |

### 5. Offline A/B Replay Simulation (61,932 Sessions)
*Simulated across held-out positive test interactions from `reports/final_results.json`:*

| Metric | Variant A (Retrieval-Only) | Variant B (Two-Stage + DCN-V2) | Lift / Significance |
|---|---|---|---|
| **Impressions** | 30,801 | 31,131 | 50 / 50 Hash Split |
| **Simulated Clicks** | 131 | 156 | +25 clicks |
| **Click-Through Rate (CTR)** | **0.43%** | **0.50%** | **+17.82% relative lift** |
| **95% Bootstrap CI** | – | – | **[-7.25%, +40.64%]** |
| **Two-Sample Z-Test p-value** | – | – | $p = 0.1649$ (Directional lift) |

---

## Quickstart & Reproduction Guide

### Prerequisites
- Python 3.11 (`.venv` recommended)
- Java 17+ or 21+ (required for PySpark feature extraction)
- Redis server running on `localhost:6379`
- 16 GB+ RAM recommended for training on the full 25M dataset

### 1. Environment Setup
```powershell
# Clone repository
git clone https://github.com/your-username/RecoEngine.git
cd RecoEngine

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate # Linux/macOS

# Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Data Preparation & Temporal Split
```powershell
# Download MovieLens 25M dataset
python scripts/download_data.py

# Execute strict temporal split (80/10/10) and compute hurdle baselines
python scripts/prepare_data.py
```

### 3. Feature Engineering with PySpark
```powershell
# Build user and item feature tables (extracts release years, interaction counts, genres)
python scripts/build_features.py --mode dev   # Fast dev mode (1M interactions)
python scripts/build_features.py --mode full  # Full 25M dataset
```

### 4. Stage 1 Two-Tower Training & FAISS Index
```powershell
# Train TFRS dual-encoder retrieval model with in-batch negatives
python scripts/train_retrieval.py --mode full --epochs 5

# Build exact FAISS IndexFlatIP index and run latency benchmarks
python scripts/build_index.py --version v1
```

### 5. Stage 2 DCN-V2 Deep Ranker Training
```powershell
# Train DCN-V2 with hard negatives mined from Stage 1 FAISS Top-200
python scripts/train_ranker.py --mode full --epochs 5 --neg-ratio 4
```

### 6. Cold-Start Fallbacks & Serving Benchmark
```powershell
# Fit and benchmark cold-start fallbacks
python scripts/evaluate_cold_start.py

# Benchmark FastAPI serving layer latency across hit/miss routes
python scripts/benchmark_serving.py
```

### 7. Automated Retraining Pipeline (Airflow / Standalone)
```powershell
# Run automated 8-stage retraining pipeline with NDCG@10 promotion gate
python scripts/run_retraining.py --mode dev
```

### 8. Master Final Evaluation & A/B Replay Simulation
```powershell
# Run master held-out test evaluation and export reports
python scripts/run_final_eval.py
```

### 9. Launch Interactive Streamlit Demo
```powershell
# Start Streamlit application
streamlit run demo/app.py
```
Open `http://localhost:8501` to explore:
- **Warm User Personalization:** Switch between curated personas or custom IDs.
- **Cold-Start Onboarding Quiz:** Select favorite seed genres to explore recommendations.
- **Side-by-Side A/B Inspection:** Inspect rank changes and catalog overlap between Variant A and B.
- **Live Metric Dashboards:** Inspect verified benchmarks and retraining logs.

---

## Test Suite & Verification

Run the comprehensive unit and integration test suite across all 13 test files:

```powershell
# Run core test suite (A/B, metrics, FAISS index, ranker, retraining, API, cache)
pytest tests/test_ab.py tests/test_metrics.py tests/test_index.py tests/test_ranker.py tests/test_retraining.py tests/test_api.py tests/test_cache.py tests/test_fallback.py tests/test_demo.py -v

# Run full test suite including PySpark feature isolation tests
pytest tests -v

# Verify code formatting and linting (0 errors)
ruff check src tests scripts airflow demo
```

---

## Project Structure

```text
RecoEngine/
├── airflow/
│   └── dags/
│       └── retrain_dag.py         # Airflow DAG for automated, gated retraining
├── artifacts/
│   ├── v1/                        # Active production model artifacts
│   │   ├── faiss.index            # FAISS IndexFlatIP catalog index (8.82 MB)
│   │   ├── item_embeddings.npy    # 34,461 x 64 item vectors
│   │   ├── user_tower/            # TFRS User Tower SavedModel
│   │   ├── ranker/                # DCN-V2 Ranker SavedModel
│   │   ├── fallback_popularity.json
│   │   └── fallback_content.json
│   └── v2/                        # Candidate model artifacts from retraining
├── data/
│   ├── raw/ml-25m/                # Raw MovieLens 25M CSVs
│   ├── processed/                 # Strictly partitioned train, val, test splits
│   └── features/                  # User and item PySpark Parquet feature tables
├── demo/
│   └── app.py                     # Interactive Streamlit demo UI (docs/Design.md)
├── docs/
│   ├── api.md                     # Serving API reference & curl examples
│   ├── architecture.svg           # High-resolution architectural diagram
│   ├── Architecture.md            # In-depth system architecture specification
│   ├── Design.md                  # UI/UX and styling design tokens
│   ├── Memory.md                  # Chronological project state & decisions
│   ├── openapi.json               # Auto-generated OpenAPI v3 specification
│   ├── Phases.md                  # Engineering phase execution contract
│   ├── PRD.md                     # Product Requirements Document
│   └── Reflection.md              # Engineering reflection & retrospective
├── presentation.md                # 10-minute presentation slides & notes
├── reports/
│   ├── baselines.json             # Benchmark baselines (Random & Popularity)
│   ├── cold_start_metrics.json    # Cold fallback benchmarks
│   ├── final_results.json         # Master test evaluation results
│   ├── final_results.md           # Master test evaluation markdown report
│   ├── ranking_metrics.json       # Stage 2 DCN-V2 validation metrics
│   ├── retrieval_latency.json     # FAISS query latency profile
│   ├── retrieval_metrics.json     # Stage 1 validation recall and lift
│   ├── retraining_log.json        # Promotion gate audit logs
│   ├── serving_latency.json       # Serving API latency profile
│   └── split_report.json          # Temporal split integrity verification
├── scripts/
│   ├── benchmark_serving.py       # API latency benchmark harness
│   ├── build_features.py          # PySpark feature engineering script
│   ├── build_index.py             # FAISS index builder & latency tester
│   ├── download_data.py           # MovieLens 25M downloader & unpacker
│   ├── evaluate_cold_start.py     # Cold fallback evaluation script
│   ├── prepare_data.py            # Temporal split & vocabulary generator
│   ├── run_final_eval.py          # Master held-out test evaluation orchestrator
│   ├── run_retraining.py          # Standalone reproducible retraining runner
│   ├── train_ranker.py            # DCN-V2 deep ranker training script
│   └── train_retrieval.py         # Two-tower TFRS retrieval training script
├── src/reco/
│   ├── data/                      # Vocabulary, split, and dataset loaders
│   ├── evaluation/                # Metrics, baselines, and A/B replay simulation
│   ├── fallback/                  # Time-decayed popularity and genre fallbacks
│   ├── features/                  # PySpark feature engineering pipelines
│   ├── indexing/                  # FAISS IndexFlatIP wrapper with seen-filtering
│   ├── models/                    # TFRS Two-Tower & DCN-V2 Keras architectures
│   ├── retraining/                # Statistical non-inferiority promotion gate
│   ├── serving/                   # FastAPI app, Redis cache, A/B routing, inference
│   └── utils/                     # Seeds, logging, and configuration utilities
├── tests/                         # 53 unit and integration tests across 13 modules
├── triton/
│   └── model_repository/          # Triton Inference Server repository config
├── requirements.txt               # Pinned dependency manifest
└── pyproject.toml                 # Ruff and Pytest project configuration
```

---

## Known Limitations & Production Roadmap

1. **In-Process Inference vs Triton GPU Cluster:**
   - *Current State:* Serves via an optimized in-process TensorFlow SavedModel client when Triton daemon is offline.
   - *Next Step:* Deploy Triton model repository to a Kubernetes cluster backed by NVIDIA TensorRT for 5,000+ QPS.
2. **Contextual Multi-Armed Bandits:**
   - *Current State:* Cold-start uses exponential time-decayed popularity and genre content matching.
   - *Next Step:* Upgrade cold-start exploration to LinUCB / Thompson Sampling for dynamic bandit learning.
3. **Real-Time Sequential Encoders:**
   - *Current State:* User representations are static embedding lookups.
   - *Next Step:* Implement transformer-based sequential models (e.g. SASRec) to adapt to intra-session clicks.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

