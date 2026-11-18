# RAG Architecture Comparison (Offline / Configured Model)

Lexical grounding is token overlap, not factual correctness. Refusals are excluded from its average.
Bootstrap nonpositive fractions are descriptive, not calibrated significance p-values.

- **Corpus Size**: 9 documents
- **Golden Evaluation Set**: 9 stratified QA pairs

## Architecture Comparison (with 95% Bootstrap Confidence Intervals)

| Architecture | Chunks | Recall@3 | Recall@5 | MRR | NDCG@5 | Context Prec@5 | Lexical grounding proxy | Refusal Acc | Latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 29 | 1.000 | 1.000 [1.000, 1.000] | 1.000 | 1.000 [1.000, 1.000] | 0.982 | 1.000 | N/A | 0.67 |
| Dense | 29 | 1.000 | 1.000 [1.000, 1.000] | 0.815 | 0.862 [0.710, 0.959] | 0.824 | 1.000 | N/A | 0.51 |
| Hybrid RRF | 29 | 1.000 | 1.000 [1.000, 1.000] | 0.944 | 0.959 [0.877, 1.000] | 0.926 | 1.000 | N/A | 0.62 |
| Hybrid RRF + feature reranker | 29 | 1.000 | 1.000 [1.000, 1.000] | 0.944 | 0.959 [0.877, 1.000] | 0.926 | 1.000 | N/A | 0.87 |

## Paired Bootstrap Difference Summary (Flagship vs Naive Fixed+Dense Baseline)

- **Recall@5 Δ**: `+0.0000` (bootstrap nonpositive fraction = `1.0000`)
- **MRR Δ**: `+-0.0556` (bootstrap nonpositive fraction = `1.0000`)
- **NDCG@5 Δ**: `+-0.0410` (bootstrap nonpositive fraction = `1.0000`)

## Category Breakdown (Flagship Configuration)

| Category | Count | Key Metrics |
| --- | ---: | --- |
| conceptual_semantic | 9 | Recall@5=1.000, MRR=0.944, NDCG@5=0.959, KW Cov=0.222 |
