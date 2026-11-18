# Data Scientist Interview & Freelance Portfolio Playbook

Use this playbook to turn `/Volumes/Data/Projects/Rag` into high-impact **resume bullet points**, **STAR behavioral stories**, **technical Q&A mastery**, and **client-ready freelance proposals**.

---

## 1. Resume Bullet Points (Tailored for Senior / Applied Data Scientist)

- **Architected an enterprise-grade Hybrid RAG platform** combining Parent-Child hierarchical chunking, Okapi BM25 + dense semantic search via Reciprocal Rank Fusion ($k=60$), and a cross-feature interaction reranker; raised **Recall@5 by +7.35 pp ($p=0.048$)** and **NDCG@5 by +5.76 pp ($p=0.046$)** across a 40-question stratified evaluation benchmark (`95%` bootstrap CIs).
- **Solved OOV single-token cosine collapse & prompt-injection vulnerabilities** by engineering vocabulary-coverage confidence damping (`min_confidence_threshold=0.22`) and PII/jailbreak guardrails, achieving **100.0% Refusal Accuracy** and **100.0% Answer Faithfulness** on out-of-scope and adversarial probes.
- **Engineered 10,000-concurrency resilience** with L1 SHA-256 exact caching, L2 vector semantic caching ($\cos \ge 0.90$), singleflight request coalescing, token-bucket rate limiting, bounded async semaphores, and tri-state circuit breakers—completing **10,000/10,000 concurrent requests (100% success) at 3,592.7 RPS** with **74.96% LLM offload**.

---

## 2. Three High-Impact STAR Stories for DS Interviews

### Story A: "Tell me about a time an ML metric looked good on paper, or a model failed silently in edge cases."
- **Situation**: Built an evaluation harness for enterprise search covering exact compliance codes (`SOC2-CC6.7`), paraphrases, multi-hop questions, and unanswerable/out-of-scope traps.
- **Task**: Ensure the RAG system refuses unanswerable questions cleanly instead of hallucinating.
- **Action**: Discovered during error analysis that out-of-scope queries containing a single in-corpus word (e.g., *"croissant ... ratio"* matching *"Liquidity Coverage Ratio"*) caused L2-normalized dense embeddings to collapse onto 1 dimension and report `0.93` cosine similarity (`16.7%` initial refusal accuracy). Implemented corpus-vocabulary and passage-token coverage damping inside confidence estimation.
- **Result**: Refusal accuracy jumped from **16.7% to 100.0%** with zero degradation on answerable queries (`Recall@5 = 0.971`).

### Story B: "How do you prove an architectural change actually improved a RAG pipeline?"
- **Situation**: Needed to justify Parent-Child + Hybrid RRF + Reranker complexity over a naive fixed-chunk Dense baseline.
- **Task**: Build a statistically defensible ablation pipeline rather than eyeballing 5 cherry-picked examples.
- **Action**: Curated a 40-query stratified golden set (`exact_lookup`, `conceptual_semantic`, `multi_hop`, `unanswerable_guardrail`), ran 6 architectures in isolation (`use_cache=False`), and computed 500-resample non-parametric bootstrap 95% CIs plus paired bootstrap p-values (`src/rag_platform/evaluation.py`).
- **Result**: Demonstrated statistically significant improvements (`Recall@5 +0.0735, p=0.0480`; `NDCG@5 +0.0576, p=0.0460`) while revealing the exact latency trade-off (`1.16 ms` vs `4.50 ms` per cold retrieval).

### Story C: "How did you scale an AI system for high concurrency and cost control?"
- **Situation**: Enterprise clients wanted 10,000 concurrent users without paying 10,000x LLM GPU costs or hitting thundering-herd crashes.
- **Task**: Design and load-test a high-concurrency pipeline that maintains sub-millisecond median latency and 100% completion.
- **Action**: Combined L1 exact caching, L2 semantic caching ($\tau=0.90$), `SingleflightCoalescer` (mapping simultaneous identical cold misses to one `asyncio.Future`), and precomputed sparse/reranker tokenization matrices (cutting single-query embedding overhead by >20x).
- **Result**: Verified **10,000/10,000 concurrent completions (3,592.7 RPS, P50 = 0.017 ms, 74.96% cache/coalescing offload)**.

---

## 3. Top 12 Data Scientist RAG Interview Questions & Model Answers

1. **Why use Reciprocal Rank Fusion (RRF) instead of normalizing Dense and BM25 scores and summing them?**
   - Cosine similarities stay tightly clustered (e.g., `0.35–0.75`), while BM25 scores are unbounded (`0–35+`). Min-max scaling per query distorts scores when top hits have flat distributions. RRF ($1 / (k + r)$, $k=60$) depends solely on rank, making fusion distribution-agnostic and robust.
2. **How do you choose the L2 Semantic Cache similarity threshold $\tau$?**
   - Treat it as a precision-recall ROC trade-off (`ML-OPS-009`). Lowering $\tau$ (e.g. `0.80`) increases hit rate but risks false cache collisions (returning an answer for `ERR_AUTH_4019` when asked about `ERR_GW_4290`). We set $\tau = 0.90$ and store scope keys (`mode || reranker || top_k || domain_filter`) to prevent cross-scope leaks.
3. **Why does Parent-Child chunking beat large fixed chunks?**
   - Large chunks suffer from vector dilution across unrelated topics; small child chunks (`260 chars`) produce sharp embeddings and BM25 matches, while returning the enclosing `900-char` parent preserves full tables, formulas, and caveats for the generator.
4. **How do you prevent a multi-hop question from filling top-5 with 5 chunks from the same document?**
   - Enforce a diversity cap (`max_per_doc=2` in `deduplicate_and_select`) and deduplicate identical parent blocks before finalizing top-K context.
5. **What is Singleflight request coalescing, and how does it differ from caching?**
   - Caching serves *subsequent* requests after the first finishes; Singleflight deduplicates *simultaneous in-flight* misses before any cache entry exists, collapsing 250 concurrent identical requests into 1 backend execution.
6. **How do you decouple Retrieval evaluation from Generation evaluation?**
   - Evaluate retrieval independently using `Recall@K`, `MRR`, `NDCG@K`, and `ContextPrecision@K` against `relevant_doc_ids`; evaluate generation using `Answer Faithfulness` (context entailment), `Citation Precision`, and `Refusal Accuracy`.
7. **Why use Bootstrap Confidence Intervals on RAG evals?**
   - Evaluation sets (40–200 queries) are small samples; a +3% point delta could be noise. Non-parametric bootstrap CIs and paired difference tests verify statistical significance (`p < 0.05`).
8. **What causes post-retrieval ACL filtering to fail in multi-tenant RAG?**
   - Post-filtering top-K results leaks existence/count hints and can leave 0 authorized results inside top-5. Always apply `tenant_id` / `domain` / `acl_groups` as pre-filters inside index traversal.
9. **How do you detect embedding drift in production?**
   - Monitor Population Stability Index (PSI) on top-1 cosine similarity distributions and KL divergence on query clusters (`ML-OPS-003`); PSI > 0.25 triggers hard-negative mining from BM25 false positives and bi-encoder fine-tuning.
10. **How do you handle upstream LLM API outages in a Tier-0 SLA service?**
    - Use a tri-state `CircuitBreaker` (`CLOSED -> OPEN -> HALF_OPEN`). When failures cross threshold (`10`), fail over gracefully to grounded extractive synthesis from reranked parent chunks instead of throwing HTTP 500s.
11. **What is the most dangerous failure mode when profiling async Python under 10k concurrency?**
    - Blocking CPU work (e.g., running regex tokenization or `sklearn.transform` validation 36x per request inside the event loop) starves concurrent coroutines and triggers queue timeouts. Pre-tokenizing static chunks and vectorizing inference arrays eliminated our cold tail bottlenecks.
12. **How do you satisfy GDPR Right-to-Erasure in immutable HNSW snapshots?**
    - Per-subject Data Encryption Keys (DEKs) stored in Vault (`SEC-COMP-003`): delete vector IDs, purge L1/L2 caches, and destroy the subject DEK (cryptographic shredding).

---

## 4. Freelance Consulting Proposal & Scoping Blueprint

When pitching enterprise RAG engagements on Upwork, Toptal, or direct B2B consulting:

### Recommended 3-Tier Package Structure
1. **Tier 1 — RAG Diagnostic & Evaluation Audit (1–2 Weeks, \$3,500–\$6,000)**
   - Deliverable: Curated 40–100 Golden QA dataset, 6-architecture ablation report with 95% CIs, failure-mode taxonomy (chunking, OOV codes, hallucination rate), and ROI roadmap.
2. **Tier 2 — Production Hybrid RAG MVP (3–5 Weeks, \$10,000–\$20,000)**
   - Deliverable: Domain-tuned ingestion pipeline (Parent-Child + PII scrubbing), Hybrid RRF + Reranker service, FastAPI SSE API + Interactive Workbench UI, and automated `pytest` regression suite.
3. **Tier 3 — High-Concurrency Enterprise Scale & Handover (6–8 Weeks, \$25,000–\$45,000+)**
   - Deliverable: Everything in Tier 2 plus Redis L1 + L2 Semantic Cache, Singleflight coalescing, Qdrant sharded Kubernetes manifests, Locust/async 10k-concurrency load test report, circuit breaker failover, and team training runbooks.
