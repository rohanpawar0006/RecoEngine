# RecoEngine Final Evaluation Report (Held-Out Test Set)

**Evaluation Date:** 2026-11-04T15:20:10Z  
**Dataset:** MovieLens 25M (temporal 80/10/10 split)  
**Total Held-Out Test Users:** 18,276 (3,996 Warm, 14,280 Cold)  

---

## 1. Executive Summary & Success Criteria Audit

| Success Criteria (PRD Section 8) | Target Hurdle | Achieved Result | Margin / Lift | Verdict |
|---|---|---|---|---|
| **SC1: Retrieval Beats Popularity** | Recall@200 > 0.11373 | **0.05617** | **+-50.61%** | **FAIL** |
| **SC2: Ranker Beats Retrieval** | NDCG@10 > 0.01215 | **0.01604** | **+32.05%** | **PASS** |
| **SC3: Cold-Start Beats Random** | NDCG@10 > 0.00198 | **0.13975** | **+6955.1%** | **PASS** |
| **SC4: Serving Latency Measured** | p95 < 150 ms | **p50: 17.6 ms, p95: 20.9 ms** | Sub-25ms | **PASS** |
| **SC5: Retraining Pipeline Logged** | Audit log with gate decision | `reports/retraining_log.json` | Both pass & rejection tested | **PASS** |

---

## 2. Warm Users: Multi-Stage Recommendation Quality

Evaluated on 2,000 warm test users:

| Model / Baseline | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 | Recall@200 | NDCG@200 |
|---|---|---|---|---|---|---|
| **Random Baseline** | 0.00011 | 0.00068 | 0.00102 | 0.00099 | 0.00329 | 0.00179 |
| **Popularity Baseline (Warm)** | 0.01413 | 0.02967 | 0.04397 | 0.03547 | 0.11373 | 0.05844 |
| **Stage 1: Two-Tower (TFRS + FAISS)** | 0.00462 | 0.01215 | **0.02001** | **0.01530** | **0.05617** | **0.02692** |
| **Stage 2: DCN-V2 Deep Ranker** | **0.00660** | **0.01604** | *–* | *–* | *–* | *–* |

> **Key Finding:** Stage 1 Two-Tower retrieval achieves **0.05617 Recall@200** (+-50.61% relative lift over the warm popularity baseline). Stage 2 DCN-V2 re-ranking lifts NDCG@10 to **0.01604** (+32.05% relative lift over retrieval-only candidate ordering).

---

## 3. Cold-Start Users: Fallback Strategies

Evaluated on 2,000 cold test users (unseen in training split):

| Fallback Strategy | Recall@10 | NDCG@10 | Recall@50 | Recall@200 |
|---|---|---|---|---|
| **Random Baseline** | 0.00026 | 0.00198 | 0.00126 | 0.00504 |
| **Raw Global Popularity** | 0.05122 | 0.29127 | 0.14831 | 0.31480 |
| **Time-Decayed Popularity Fallback** | **0.02490** | **0.13975** | **0.05782** | **0.08245** |
| **Content-Based Fallback (Seed Genre)** | 0.01150 | 0.07630 | 0.03773 | 0.09062 |

---

## 4. A/B Replay Simulation Results

Offline replay simulation using 61,932 held-out test interactions as simulated clicks:

| Metric | Variant A (Retrieval-Only) | Variant B (Two-Stage + DCN-V2) | Comparison / Lift |
|---|---|---|---|
| **Impressions** | 30,801 | 31,131 | 50 / 50 Hash Split |
| **Simulated Clicks** | 131 | 156 | – |
| **Click-Through Rate (CTR)** | **0.0043** | **0.0050** | **+17.82%** relative lift |
| **Absolute Difference** | – | – | **+0.00076** |
| **95% Bootstrap CI (Relative)** | – | – | **[-7.25%, +40.64%]** |
| **p-value (Two-Tailed)** | – | – | **0.164932** (Directionally positive, not statistically significant at alpha=0.05) |

---

## 5. End-to-End Latency Profile (from `reports/serving_latency.json`)

| Path | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---|---|---|
| **Warm Cache Hit (Redis)** | 6.00 ms | **5.62 ms** | 9.30 ms | 10.07 ms |
| **Warm Full Pipeline (Miss)** | 18.81 ms | **17.65 ms** | 20.93 ms | 43.87 ms |
| **Cold Popularity Fallback** | 6.50 ms | **6.03 ms** | 9.49 ms | 10.63 ms |
| **Cold Content-Based Fallback** | 163.21 ms | **159.24 ms** | 196.69 ms | 224.38 ms |
