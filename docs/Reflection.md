# Engineering Reflection: RecoEngine Internship
**Author:** Rohan | **Role:** AI/ML Engineering Intern  
**Reporting Manager:** Rishi | **Project:** RecoEngine — Two-Stage Industrial Recommender  
**Date:** November 2026 | **Context:** 1-Month Milestone Internship Deliverable  

---

## 1. Executive Summary & Overview

Over the course of this four-week internship, my mandate was to design, implement, evaluate, and productionize a two-stage recommendation system on the full MovieLens 25M benchmark dataset (25,000,095 ratings, 162,541 users, 59,047 items).

Rather than treating recommendation solely as an offline metric-optimization task in a Jupyter notebook, I approached RecoEngine as a production-grade distributed system. The resulting codebase spans data engineering (PySpark feature pipelines), vector retrieval (TFRS dual-encoder and FAISS exact inner product search), deep CTR ranking (DCN-V2 with polynomial cross layers), low-latency web serving (FastAPI, Redis fail-open caching, Triton model repository), automated MLOps (Airflow retraining DAG with statistical promotion gating), and offline experimentation (salted hash A/B testing with bootstrap confidence intervals).

Every reported metric and latency figure is verifiable from scripts and machine-readable JSON reports in `reports/`, evaluated on a strictly held-out temporal test split.

---

## 2. Core Architectural Decisions & Ownership

As the primary engineer on this project, I owned the architectural roadmap, design contracts, and implementation from scratch:

1. **Strict Temporal Partitioning (80/10/10):**
   - *Decision:* Sorted all 25M interactions chronologically and partitioned at strict timestamp boundaries ($t_1 = 1,489,520,387$, $t_2 = 1,535,000,105$).
   - *Rationale:* Random train/test splits cause severe future-to-past data leakage. Users in production cannot be recommended items using interactions they had in the future.
   - *Impact:* Protected our evaluation from metric inflation and exposed the true cold-start challenge (78.1% of test users had zero prior training interactions).

2. **Feature Store Isolation with PyArrow:**
   - *Decision:* Engineered user and item feature tables using PySpark with parquet output written via PyArrow.
   - *Rationale:* Avoided native Hadoop `winutils.exe` NTFS permission traps on Windows while enforcing strict leakage isolation (features computed purely from the 20M train split).

3. **Popularity-Prior Calibration for Two-Tower Retrieval:**
   - *Decision:* Added a frequency-calibrated prior ($0.4 \times \log(1 + \text{item\_pop})$) to raw dot-product similarity scores in Stage 1 candidate generation.
   - *Rationale:* In-batch negative sampling penalizes popular items during training. Without correction, raw inner product search suffers from hubness and neglects catalog staples.
   - *Impact:* Lifted validation Recall@200 from 0.056 to **0.25647**, beating the warm popularity baseline (**0.17418**) by **+47.2%**.

4. **Hard Negative Mining for Stage 2 Deep Ranker:**
   - *Decision:* Constructed ranker training datasets by mining negative candidates from Stage 1 FAISS Top-200 outputs at a 1:4 positive-to-negative ratio.
   - *Rationale:* Training a ranker on uniform random negatives creates a severe training-serving skew. The ranker in production only ever sees items that already passed Stage 1 retrieval.
   - *Impact:* Achieved **+32.05% relative NDCG@10 lift** on the held-out test set over Stage 1 candidate ordering (0.01604 vs 0.01215).

5. **Exponential Recency Decay for Cold-Start Fallbacks:**
   - *Decision:* Implemented a time-decayed popularity model with a 2-year half-life ($e^{-\lambda \Delta t}$) alongside a genre content-based matching engine.
   - *Rationale:* Global popularity counts over 20 years over-index on blockbusters from the 1990s (e.g. *Pulp Fiction*, *Shawshank Redemption*), which fails modern users.
   - *Impact:* Achieved **0.13975 NDCG@10** on cold test users—a **+6,955% lift** over random and **+71% lift** over raw static popularity.

6. **Dual-Mode Serving & Fail-Open Caching:**
   - *Decision:* Designed `RankerInference` client to query Triton Inference Server when available, falling back seamlessly to an in-process TensorFlow SavedModel when offline. Built Redis caching to fail open on network errors.
   - *Impact:* Guaranteed zero-downtime serving across both containerized Linux and native Windows host environments while maintaining sub-20ms p50 latency.

7. **Statistical Promotion Gating in Airflow:**
   - *Decision:* Created an automated 8-stage retraining DAG with a non-inferiority gate requiring candidate NDCG@10 to be within tolerance ($\tau = 0.001$) of production before swapping model weights and invalidating the Redis cache.
   - *Impact:* Demonstrated both successful promotion to `v2` and automated blocking of degraded candidates in unit tests (`reports/retraining_log.json`).

---

## 3. Engineering Challenges & Solutions

### Challenge 1: The "Hurdle Baseline" Trap
Early in the project, when comparing our Stage 1 two-tower model against a global popularity baseline across the entire validation set, the raw popularity numbers appeared unusually high. Investigating the demographic split revealed that 70.6% of validation users were cold start. Global popularity was capturing general trends for cold users, skewing the overall comparison.
- **Solution:** I separated the evaluation into distinct warm-user and cold-user cohorts. Warm users were benchmarked against warm popularity (0.17418) where the two-tower model achieved 0.25647 (+47.2% lift). Cold users were benchmarked separately against cold fallbacks. This established methodological rigor and prevented deceptive aggregate metrics.

### Challenge 2: Training-Serving Distribution Mismatch in Deep Ranking
Initially, ranking models trained on randomly sampled catalog negatives achieved high training AUC (>0.95), but provided negligible re-ranking lift when applied to top-200 retrieved candidates.
- **Solution:** I diagnosed this as distribution mismatch: random negatives are trivially distinguishable from true positives (e.g., comparing a Sci-Fi blockbuster to an obscure 1930s foreign documentary). At serving time, however, the ranker must distinguish between 200 items that *all* look like Sci-Fi blockbusters. By mining hard negatives directly from Stage 1 FAISS outputs, the ranker learned the subtle feature interactions necessary to improve NDCG@10 by +32.05%.

### Challenge 3: Windows Native MLOps Dependencies
Deploying industrial MLOps tools (Apache Spark, Redis, Triton, and Airflow) natively on a Windows development environment presents notorious compatibility hurdles—notably Hadoop `winutils.exe` native library errors, Triton Docker requirements, and Airflow Unix process-forking limitations.
- **Solution:** Rather than cutting features or claiming environment blockers:
  - Configured PySpark to write Parquet via PyArrow, bypassing Hadoop native binaries entirely.
  - Deployed Redis 8.10 via native Windows daemon.
  - Built a dual-mode serving abstraction that prepares the Triton `config.pbtxt` repository structure while executing via an optimized in-process SavedModel engine.
  - Built an Airflow DAG with modular TaskFlow operators, paired with a sequential standalone runner (`scripts/run_retraining.py`) to execute full retrain cycles natively.

---

## 4. Key Takeaways & Lessons Learned

1. **Production ML is 20% Modeling, 80% Systems Engineering:**
   The hardest and most impactful parts of building RecoEngine were not tweaking hyperparameters in TensorFlow, but constructing non-leaking feature pipelines, designing low-latency vector index queries, structuring cache invalidation logic, and implementing automated gates that prevent bad models from reaching production.

2. **Honesty in Evaluation Builds Engineering Credibility:**
   It is tempting to mask cold users, use random splits, or report uncalibrated lifts. By holding ourselves to strict temporal boundaries, measuring baselines first, and calculating 95% bootstrap confidence intervals for A/B simulations, we generated results that engineering leadership can trust.

3. **Defensive, Fail-Open Design is Non-Negotiable:**
   In a production recommendation pipeline, dependencies like Redis or Triton will experience transient latency spikes or network partitions. Building graceful degradation paths ensured that our `/recommend` endpoint never returns a 500 error to the client, falling back transparently from cache to model, and from model to cold-start heuristics.

---

## 5. Future Roadmap & Production Next Steps

If granted another month on this project, I would prioritize the following extensions:
1. **GPU-Backed Triton Deployment:** Deploy the Triton model repository onto a dedicated Linux GPU cluster (Kubernetes with KServe) to benchmark batch inference throughput at 5,000+ QPS.
2. **Sequential Session Modeling:** Replace static user embeddings with a transformer-based sequential recommendation model (e.g. SASRec or BST) to capture intra-session user intent in real time.
3. **Multi-Armed Bandits for Cold-Start Exploration:** Upgrade the static genre-matching fallback to a contextual multi-armed bandit (LinUCB or Thompson Sampling) to optimize the exploration-exploitation trade-off for new users.
4. **Online Feature Streaming:** Integrate Apache Kafka and Feast to stream real-time interaction counts and last-seen items into the Redis feature store with sub-second freshness.

---

## 6. Acknowledgments

I want to express my gratitude to my manager Rishi for setting rigorous industrial standards from Day 1, emphasizing temporal integrity, and encouraging me to tackle the complete end-to-end stack. This project has provided an invaluable foundation in real-world recommendation systems engineering.

