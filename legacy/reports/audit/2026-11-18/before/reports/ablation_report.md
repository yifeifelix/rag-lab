# Enterprise RAG Data Science Ablation Report

- **Corpus Size**: 65 enterprise documents
- **Golden Evaluation Set**: 40 stratified QA pairs

## Architecture Comparison (with 95% Bootstrap Confidence Intervals)

| Architecture | Chunks | Recall@3 | Recall@5 | MRR | NDCG@5 | Context Prec@5 | Faithfulness | Refusal Acc | Latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed Chunk + Naive Dense | 216 | 0.897 | 0.897 [0.794, 0.985] | 0.892 | 0.879 [0.773, 0.968] | 0.869 | 1.000 | 1.000 | 0.34 |
| Recursive MD + Dense | 142 | 0.902 | 0.902 [0.794, 0.990] | 0.858 | 0.862 [0.756, 0.958] | 0.839 | 1.000 | 1.000 | 0.43 |
| Recursive MD + Okapi BM25 | 142 | 0.956 | 0.956 [0.882, 1.000] | 0.951 | 0.940 [0.870, 0.995] | 0.916 | 1.000 | 1.000 | 0.48 |
| Semantic Breakpoint + Hybrid RRF | 130 | 0.941 | 0.971 [0.912, 1.000] | 0.899 | 0.910 [0.830, 0.978] | 0.858 | 1.000 | 1.000 | 0.52 |
| Parent-Child + Hybrid RRF | 281 | 0.868 | 0.971 [0.912, 1.000] | 0.872 | 0.889 [0.805, 0.968] | 0.856 | 1.000 | 1.000 | 0.75 |
| Parent-Child + Hybrid RRF + Reranker | 281 | 0.926 | 0.971 [0.912, 1.000] | 0.934 | 0.936 [0.863, 0.995] | 0.920 | 1.000 | 1.000 | 2.00 |

## Paired Significance Test (Flagship vs Naive Fixed+Dense Baseline)

- **Recall@5 Δ**: `+0.0735` (p-value = `0.0480`)
- **MRR Δ**: `+0.0417` (p-value = `0.1300`)
- **NDCG@5 Δ**: `+0.0576` (p-value = `0.0460`)

## Category Breakdown (Flagship Configuration)

| Category | Count | Key Metrics |
| --- | ---: | --- |
| exact_lookup | 12 | Recall@5=1.000, MRR=1.000, NDCG@5=1.000, KW Cov=0.792 |
| conceptual_semantic | 12 | Recall@5=0.917, MRR=0.812, NDCG@5=0.839, KW Cov=0.208 |
| multi_hop | 10 | Recall@5=1.000, MRR=1.000, NDCG@5=0.977, KW Cov=0.425 |
| unanswerable_guardrail | 6 | Refusal Acc=1.000, Faithfulness=1.000 |
