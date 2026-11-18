# Enterprise 10k-Concurrency RAG Platform & Data Science Workbench

A production-architected, statistically benchmarked **Enterprise Retrieval-Augmented Generation (RAG)** platform and **Data Science Evaluation Lab**, designed for senior DS job interviews, freelance client demos, and hands-on system mastery.

All virtual environments, caches, datasets, and reports reside strictly under `/Volumes/Data/Projects/Rag`.

---

## Key Capabilities & Verified Metrics

### 1. Data Science Ablation Benchmark (`40` Stratified Golden QA Pairs, `65` Enterprise Docs)
Evaluated across 4 categories (`exact_lookup`, `conceptual_semantic`, `multi_hop`, `unanswerable_guardrail`) with **500x Bootstrap 95% CIs** (`reports/ablation_report.md`):

| Architecture | Chunks | Recall@5 | MRR | NDCG@5 | Faithfulness | Refusal Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed Chunk + Naive Dense (Baseline) | 216 | 0.897 | 0.892 | 0.879 | 1.000 | 1.000 |
| Recursive MD + Dense | 142 | 0.902 | 0.858 | 0.862 | 1.000 | 1.000 |
| Recursive MD + Okapi BM25 | 142 | 0.956 | 0.951 | 0.940 | 1.000 | 1.000 |
| Semantic Breakpoint + Hybrid RRF | 130 | 0.971 | 0.899 | 0.910 | 1.000 | 1.000 |
| Parent-Child + Hybrid RRF | 281 | 0.971 | 0.872 | 0.889 | 1.000 | 1.000 |
| **Parent-Child + Hybrid RRF + Reranker (Flagship)** | **281** | **0.971** | **0.934** | **0.936** | **1.000** | **1.000** |

- **Paired Significance vs. Naive Baseline**: `Recall@5 +0.0735 (p = 0.0480)`, `NDCG@5 +0.0576 (p = 0.0460)`.

### 2. 10,000-Concurrency Load Benchmark (`reports/benchmark_10k_report.md`)
Simulated Zipf enterprise workload (`35%` L1 Exact, `10%` Simultaneous Viral Burst, `30%` L2 Semantic Paraphrase, `25%` Cold Tail RAG):
- **Completion**: **`10,000 / 10,000` (`100.00%` success rate)** in **`2.783 s`** (**`3,592.7 RPS`**)
- **Overall P50 Latency**: **`0.017 ms`**
- **Cache + Singleflight Offload**: **`74.96%`** (`3,500` L1 hits, `3,000` L2 semantic hits, `996` singleflight coalesced hits)
- **Peak Backend Gate Concurrency**: **`256`** (protected by `AsyncConcurrencyGate` + `TokenBucketRateLimiter` + `CircuitBreaker`)

---

## Project Structure

```text
/Volumes/Data/Projects/Rag/
├── data/
│   ├── corpus/documents.jsonl         # 65 multi-section docs across 5 enterprise domains
│   └── eval/golden_qa.jsonl           # 40 golden QA pairs + paraphrase variants
├── deploy/
│   ├── docker-compose.yml             # FastAPI + Redis + Qdrant + Prometheus stack
│   ├── k8s-production-10k.yaml        # K8s Deployment, HPA (8-48 pods), PDB
│   └── locustfile.py                  # Distributed Locust Zipf load test
├── docs/
│   ├── 01_DS_RAG_LEARNING_ROADMAP.md          # 6-phase Data Scientist curriculum
│   ├── 02_ARCHITECTURE_10K_CONCURRENCY.md     # Little's Law & scaling blueprint
│   └── 03_INTERVIEW_AND_FREELANCE_PLAYBOOK.md # Resume bullets, STAR stories, 12 Q&As
├── reports/
│   ├── ablation_report.md / .json     # 6-config DS evaluation & bootstrap CIs
│   └── benchmark_10k_report.md / .json# 10,000-concurrency stress test telemetry
├── scripts/
│   ├── run_ablation_study.py          # Run full DS ablation study
│   └── run_10k_benchmark.py           # Run 10k concurrent async load test
├── src/rag_platform/
│   ├── benchmark.py                   # 10k-concurrency Zipf benchmark harness
│   ├── cache.py                       # L1 Exact, L2 Semantic Vector, Singleflight
│   ├── chunking.py                    # Fixed, Recursive, Semantic, Parent-Child
│   ├── config.py                      # Central RAGConfig dataclass
│   ├── data_gen.py                    # Deterministic corpus & golden QA generator
│   ├── embeddings.py                  # Fast TF-IDF+SVD embedder + OpenAI/Gemini adapters
│   ├── evaluation.py                  # Recall@K, MRR, NDCG@5, Faithfulness, Bootstrap CIs
│   ├── pipeline.py                    # End-to-end async EnterpriseRAGPipeline
│   ├── reranker.py                    # Pre-tokenized CrossFeatureReranker & diversity selector
│   ├── resilience.py                  # RateLimiter, AsyncConcurrencyGate, CircuitBreaker
│   ├── retrieval.py                   # DenseVectorIndex, CSR BM25Index, HybridRRFRetriever
│   ├── server.py                      # FastAPI REST + SSE streaming server
│   └── static/index.html              # Interactive 4-tab Web Workbench UI
└── tests/
    └── test_platform.py               # Automated pytest suite
```

---

## Quickstart Commands

```bash
cd /Volumes/Data/Projects/Rag
source .venv/bin/activate

# 1. Start the Interactive RAG & DS Workbench (http://localhost:8000)
uvicorn rag_platform.server:app --host 0.0.0.0 --port 8000

# 2. Re-run the 6-Architecture Data Science Ablation Study
python scripts/run_ablation_study.py

# 3. Re-run the 10,000-Concurrency Load Benchmark
python scripts/run_10k_benchmark.py --requests 10000

# 4. Run full automated verification suite
pytest -v
```
