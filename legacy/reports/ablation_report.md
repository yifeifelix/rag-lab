# RAG Architecture Comparison (Offline / Configured Model)

Lexical grounding is token overlap, not factual correctness. Refusals are excluded from its average.
Bootstrap nonpositive fractions are descriptive, not calibrated significance p-values.

- **Corpus Size**: 65 documents
- **Golden Evaluation Set**: 40 stratified QA pairs

## Architecture Comparison (with 95% Bootstrap Confidence Intervals)

| Architecture | Chunks | Recall@3 | Recall@5 | MRR | NDCG@5 | Context Prec@5 | Lexical grounding proxy | Refusal Acc | Latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed Chunk + Naive Dense | 216 | 0.897 | 0.897 [0.794, 0.985] | 0.892 | 0.879 [0.773, 0.968] | 0.869 | 1.000 | 1.000 | 0.38 |
| Recursive MD + Dense | 142 | 0.902 | 0.902 [0.794, 0.990] | 0.858 | 0.862 [0.756, 0.958] | 0.839 | 1.000 | 1.000 | 0.46 |
| Recursive MD + Okapi BM25 | 142 | 0.956 | 0.956 [0.882, 1.000] | 0.951 | 0.940 [0.870, 0.995] | 0.916 | 1.000 | 1.000 | 0.52 |
| Semantic Breakpoint + Hybrid RRF | 130 | 0.941 | 0.971 [0.912, 1.000] | 0.899 | 0.910 [0.830, 0.978] | 0.858 | 1.000 | 1.000 | 0.56 |
| Parent-Child + Hybrid RRF | 281 | 0.868 | 0.971 [0.912, 1.000] | 0.872 | 0.889 [0.805, 0.968] | 0.856 | 1.000 | 1.000 | 0.81 |
| Parent-Child + Hybrid RRF + Reranker | 281 | 0.926 | 0.971 [0.912, 1.000] | 0.934 | 0.936 [0.863, 0.995] | 0.920 | 1.000 | 1.000 | 1.62 |
| Awesome-RAG: LightRAG + CRAG + Reranker | 281 | 0.985 | 1.000 [1.000, 1.000] | 1.000 | 0.992 [0.983, 1.000] | 0.968 | 1.000 | 1.000 | 2.67 |

## Paired Bootstrap Difference Summary (Flagship vs Naive Fixed+Dense Baseline)

- **Recall@5 Δ**: `+0.1029` (bootstrap nonpositive fraction = `0.0170`)
- **MRR Δ**: `+0.1078` (bootstrap nonpositive fraction = `0.0170`)
- **NDCG@5 Δ**: `+0.1131` (bootstrap nonpositive fraction = `0.0040`)

## Category Breakdown (Flagship Configuration)

| Category | Count | Key Metrics |
| --- | ---: | --- |
| exact_lookup | 12 | Recall@5=1.000, MRR=1.000, NDCG@5=1.000, KW Cov=0.792 |
| conceptual_semantic | 12 | Recall@5=1.000, MRR=1.000, NDCG@5=1.000, KW Cov=0.292 |
| multi_hop | 10 | Recall@5=1.000, MRR=1.000, NDCG@5=0.972, KW Cov=0.425 |
| unanswerable_guardrail | 6 | Refusal Acc=1.000, Lexical grounding proxy=N/A |
