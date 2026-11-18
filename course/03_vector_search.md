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

# 3 · Vector search

**Time:** 40 minutes · **You will:** build a vector index with NumPy and write the top-k search yourself.

## An index is just a matrix

After embedding, our chunks are a matrix with one row per chunk. Searching means:

1. embed the question → vector `q` (384 numbers);
2. score every chunk: `scores = vectors @ q` (cosine, because everything is normalised);
3. return the `k` chunks with the highest scores.

```{code-cell} ipython3
import numpy as np
from raglab import load_documents, chunk_documents
from raglab.embeddings import get_embedder

docs = load_documents("python_docs")
chunks = chunk_documents(docs, method="recursive", max_chars=800)
embedder = get_embedder()
vectors = embedder.embed_passages([c.text for c in chunks])   # cached after the first run
print(vectors.shape, "| memory:", round(vectors.nbytes / 1e6, 2), "MB")
```

```{code-cell} ipython3
q = embedder.embed_query("How do I parse command-line arguments?")
scores = vectors @ q
print(scores.shape, scores.min().round(3), scores.max().round(3))
```

### ✏️ Exercise 3.1 - top-k

Return the **indices** of the `k` highest scores, best first, as a list of ints. If `k` is larger
than the number of scores, return all of them. Hint: `np.argsort` sorts ascending.

```{code-cell} ipython3
def top_k(scores: np.ndarray, k: int) -> list[int]:
    # YOUR CODE HERE
    raise NotImplementedError
```

```{code-cell} ipython3
from raglab.checks import check_top_k
check_top_k(top_k)
```

Use it:

```{code-cell} ipython3
try:
    for i in top_k(scores, 3):
        print(f"{scores[i]:.3f}  {chunks[i].chunk_id:28}  {chunks[i].text[:90]!r}")
except NotImplementedError:
    print("Finish exercise 3.1 first!")
```

## The library version

`raglab.retrieval.VectorRetriever` wraps exactly these steps (open `src/raglab/retrieval.py` and
read it - it's ~20 lines). Every retriever in this course has the same method:
`search(query, k) -> list[Hit]`, so they are interchangeable.

```{code-cell} ipython3
from raglab.retrieval import VectorRetriever
from raglab.display import hits_table

retriever = VectorRetriever(chunks)
hits_table(retriever.search("How can I run code after a delay without blocking?", k=5))
```

## How good is it? A first look

Our question set `qa.jsonl` lists, for each question, which document contains the answer.
A retrieval "hit" means the right document appears in the top 5:

```{code-cell} ipython3
from raglab import load_qa
qa = [q for q in load_qa("python_docs") if q["relevant_doc_ids"]]
hits = 0
for item in qa:
    found = {h.doc_id for h in retriever.search(item["question"], k=5)}
    ok = bool(found & set(item["relevant_doc_ids"]))
    hits += ok
    if not ok:
        print("MISS:", item["question"])
print(f"\n{hits}/{len(qa)} questions have the right document in the top 5")
```

**Read the misses.** Are they exact names? Multi-part questions? This habit - *look at failures
before averaging numbers* - is the most valuable skill in RAG work.

## What about millions of chunks?

Scoring every row (*brute force*) is exact and fine up to ~1 million vectors on a laptop. Beyond
that, vector databases (FAISS, Qdrant, Milvus, pgvector...) use **approximate nearest neighbour**
indexes such as HNSW: a graph you can walk to the nearest vectors in milliseconds, trading a
little recall for a lot of speed. The interface stays the same: vector in, top-k out.

### ✏️ Exercise 3.2 - chunk size vs. hit rate

Re-run the evaluation above with `max_chars=300` and `max_chars=1500`. Write down the hit counts.
Which works best here, and why might the answer differ for another corpus?

```{code-cell} ipython3
# Your experiment here
```

## Key takeaways

- Vector search = matrix-vector product + top-k. Nothing more mysterious.
- Normalise once at indexing time, then cosine is a dot product.
- Always read the misses.

## Going further

- RAG_Techniques [simple_rag](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/simple_rag.ipynb) and [local_rag_huggingface_faiss](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/local_rag_huggingface_faiss.ipynb) do the same with FAISS · `references/NirDiamant_RAG_Techniques/all_rag_techniques/`
- all-in-rag (Chinese), chapter 3 covers vector databases: `references/datawhalechina_all-in-rag/docs/chapter3/`
