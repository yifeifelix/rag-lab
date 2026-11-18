---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 6 · Re-ranking with a cross-encoder

**Time:** 35 minutes · **You will:** add a second, more accurate stage to the retriever and measure the gain and the cost.

## Two kinds of models

```text
BI-ENCODER (embeddings, lessons 2-3)          CROSS-ENCODER (this lesson)
 question ─► [model] ─► vector ┐               question + passage ─► [model] ─► relevance score
 passage  ─► [model] ─► vector ┴─► cosine
 ✔ passage vectors computed once, offline       ✔ reads both texts together → much more accurate
 ✔ search millions in milliseconds              ✘ must run once per (question, passage) pair
```

A bi-encoder never lets question and passage "see" each other. A cross-encoder does, so it can
notice that *"which function does **not** block"* contradicts a passage about a blocking call.
But scoring 300 chunks per question would be slow. So we use a **two-stage pipeline**:

1. cheap retriever (hybrid) → 30 candidates (high recall);
2. cross-encoder re-ranks those 30 → keep the best 5 (high precision).

## Try it

We use `cross-encoder/ms-marco-MiniLM-L-6-v2`, trained on Bing search queries.

```{code-cell} ipython3
from raglab import load_documents, chunk_documents
from raglab.retrieval import BM25Retriever, VectorRetriever, HybridRetriever
from raglab.rerank import CrossEncoderReranker, RerankingRetriever
from raglab.display import hits_table

chunks = chunk_documents(load_documents("python_docs"), "recursive", max_chars=800)
hybrid = HybridRetriever([BM25Retriever(chunks), VectorRetriever(chunks)])
reranker = CrossEncoderReranker()

q = "How can I make a function remember results of previous calls?"
candidates = hybrid.search(q, 30)
print("BEFORE re-ranking")
display(hits_table(candidates[:5]))
print("AFTER re-ranking")
display(hits_table(reranker.rerank(q, candidates, 5)))
```

Cross-encoder scores are **logits**: only their order matters, and they can be negative.

## Measure it

```{code-cell} ipython3
import pandas as pd
from raglab import load_qa
from raglab.evaluation import evaluate_retriever, summarize

qa = load_qa("python_docs")
reranked = RerankingRetriever(hybrid, reranker, candidates=30)
results = pd.concat([evaluate_retriever(hybrid, qa, 5, "hybrid"),
                     evaluate_retriever(reranked, qa, 5, "hybrid + rerank")])
summarize(results)
```

Look at the `ms` column too: quality costs latency. In production you'd check this against your
latency budget, and possibly run the re-ranker on a GPU.

### ✏️ Exercise 6.1 - how many candidates?

Re-ranking can only reorder what the first stage found. Evaluate `RerankingRetriever` with
`candidates=5, 10, 30, 60`. Plot MRR and average milliseconds against the number of candidates.
Where do you stop gaining?

```{code-cell} ipython3
# Your experiment here
```

### ✏️ Exercise 6.2 - when does re-ranking hurt?

Find questions where `hybrid + rerank` has a **lower** MRR than `hybrid`. Read the chunks.
Hypothesis: the model was trained on web search queries - what kind of text might confuse it?

```{code-cell} ipython3
wide = results.pivot_table(index="question", columns="system", values="mrr")
wide[wide["hybrid + rerank"] < wide["hybrid"]]
```

## Key takeaways

- Retrieve wide and cheap, then re-rank narrow and expensive.
- Re-ranking improves precision at the top; it cannot fix recall from stage 1.
- Always measure the latency cost alongside the quality gain.

## Going further

- RAG_Techniques [reranking](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/reranking.ipynb) (cross-encoder and LLM-based re-ranking) · `references/NirDiamant_RAG_Techniques/all_rag_techniques/reranking.ipynb`
- The RAG-Challenge-2 winner's write-up shows LLM re-ranking used in a competition: `references/IlyaRice_RAG-Challenge-2/README.md`
