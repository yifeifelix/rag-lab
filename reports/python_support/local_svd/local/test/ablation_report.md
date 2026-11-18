# RAG Architecture Comparison (Offline / Configured Model)

Lexical grounding is token overlap, not factual correctness. Refusals are excluded from its average.
Bootstrap nonpositive fractions are descriptive, not calibrated significance p-values.

- **Corpus Size**: 9 documents
- **Golden Evaluation Set**: 13 stratified QA pairs

## Architecture Comparison (with 95% Bootstrap Confidence Intervals)

| Architecture | Chunks | Recall@3 | Recall@5 | MRR | NDCG@5 | Context Prec@5 | Lexical grounding proxy | Refusal Acc | Latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 29 | 1.000 | 1.000 [1.000, 1.000] | 1.000 | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 1.000 | 0.55 |
| Dense | 29 | 1.000 | 1.000 [1.000, 1.000] | 0.955 | 0.966 [0.899, 1.000] | 0.955 | 1.000 | 1.000 | 0.47 |
| Hybrid RRF | 29 | 1.000 | 1.000 [1.000, 1.000] | 1.000 | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 1.000 | 0.52 |
| Hybrid RRF + feature reranker | 29 | 1.000 | 1.000 [1.000, 1.000] | 1.000 | 1.000 [1.000, 1.000] | 1.000 | 1.000 | 1.000 | 0.74 |

## Paired Bootstrap Difference Summary (Flagship vs Naive Fixed+Dense Baseline)

- **Recall@5 Δ**: `+0.0000` (bootstrap nonpositive fraction = `1.0000`)
- **MRR Δ**: `+0.0000` (bootstrap nonpositive fraction = `1.0000`)
- **NDCG@5 Δ**: `+0.0000` (bootstrap nonpositive fraction = `1.0000`)

## Category Breakdown (Flagship Configuration)

| Category | Count | Key Metrics |
| --- | ---: | --- |
| conceptual_semantic | 11 | Recall@5=1.000, MRR=1.000, NDCG@5=1.000, KW Cov=0.318 |
| unanswerable_guardrail | 2 | Refusal Acc=1.000, Lexical grounding proxy=N/A |
