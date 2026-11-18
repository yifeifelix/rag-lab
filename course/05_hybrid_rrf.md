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

# 5 · Hybrid search with Reciprocal Rank Fusion

**Time:** 30 minutes · **You will:** implement RRF and combine BM25 with vector search.

## Why not just add the scores?

```{code-cell} ipython3
from raglab import load_documents, chunk_documents
from raglab.retrieval import BM25Retriever, VectorRetriever

chunks = chunk_documents(load_documents("python_docs"), "recursive", max_chars=800)
bm25, vec = BM25Retriever(chunks), VectorRetriever(chunks)
q = "How do I remove a directory tree?"
print("BM25 scores  :", [round(h.score, 2) for h in bm25.search(q, 5)])
print("cosine scores:", [round(h.score, 2) for h in vec.search(q, 5)])
```

BM25 scores are unbounded (10, 20, 35...) while cosines live in roughly 0.3-0.9. Adding them
lets BM25 dominate; normalising them is fragile because the scales shift from query to query.

## The trick: use ranks, not scores

**Reciprocal Rank Fusion** ignores the scores and only looks at *positions*. Each ranked list
gives every item a vote of `1 / (k + rank)`; votes are summed:

```text
              BM25 rank   vector rank   RRF score (k=60)
chunk A           1            3        1/61 + 1/63 = 0.0323
chunk B           2            -        1/62        = 0.0161
chunk C           -            1        1/61        = 0.0164
```

Items that **both** methods like float to the top. The constant `k` (60 is the classic value)
softens the difference between rank 1 and rank 2.

### ✏️ Exercise 5.1 - implement RRF

`rankings` is a list of ranked lists of IDs (best first). Return a dict `id -> fused score`.

```{code-cell} ipython3
def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    # YOUR CODE HERE
    raise NotImplementedError
```

```{code-cell} ipython3
from raglab.checks import check_rrf
check_rrf(reciprocal_rank_fusion)
```

## Hybrid retriever

`HybridRetriever` asks each retriever for 30 candidates, fuses the rankings and keeps the top k:

```{code-cell} ipython3
from raglab.retrieval import HybridRetriever
from raglab.display import hits_table

hybrid = HybridRetriever([bm25, vec])
hits_table(hybrid.search(q, 5))
```

## Did it help? Quick comparison

```{code-cell} ipython3
from raglab import load_qa
from raglab.evaluation import evaluate_retriever, summarize
import pandas as pd

qa = load_qa("python_docs")
results = pd.concat([evaluate_retriever(r, qa, k=5, name=n)
                     for n, r in [("bm25", bm25), ("vector", vec), ("hybrid", hybrid)]])
summarize(results)
```

(Lesson 8 explains these metrics properly - for now, higher is better.)

Find the questions where hybrid beats **both** single methods, and where it is worse than the best one:

```{code-cell} ipython3
wide = results.pivot_table(index="question", columns="system", values="mrr")
wide[(wide["hybrid"] != wide[["bm25", "vector"]].max(axis=1))]
```

### ✏️ Exercise 5.2 - weighting

Sometimes one retriever deserves more trust. Change your `reciprocal_rank_fusion` (or write a new
function) to accept a `weights` list, so each list's vote is `weight / (k + rank)`. Try weights
like `[0.3, 0.7]` and `[0.7, 0.3]` with `HybridRetriever`-style code. Does either beat equal weights
on this question set? Be careful: tuning weights on the same questions you report is cheating
(see lesson 8, "dev vs. test").

```{code-cell} ipython3
# Your experiment here
```

## Key takeaways

- Don't add scores from different systems; fuse ranks.
- RRF is simple, robust and needs no training - the standard way to do hybrid search.
- Hybrid usually isn't the best on every question, but it is rarely terrible: a safe default.

## Going further

- RAG_Techniques [fusion_retrieval](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/fusion_retrieval.ipynb) · `references/NirDiamant_RAG_Techniques/all_rag_techniques/`
- RAG from scratch, videos 5-9 (multi-query + RAG-Fusion, which also uses RRF): `references/langchain-ai_rag-from-scratch/rag_from_scratch_5_to_9.ipynb`
- Original paper: Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods* (SIGIR 2009).
