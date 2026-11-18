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

# 2 · Embeddings: meaning as numbers

**Time:** 45 minutes · **You will:** implement cosine similarity and see why embeddings find answers that keyword search misses.

## From words to vectors - the naive way

Computers compare numbers, not meanings. The simplest way to turn text into numbers is to
**count words**. Each sentence becomes a vector with one position per vocabulary word:

```{code-cell} ipython3
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

sentences = ["the cat sat on the mat", "a kitten rested on a rug", "the stock market fell today"]
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(sentences).toarray()
print(vectorizer.get_feature_names_out())
print(counts)
```

Sentences 1 and 2 mean nearly the same thing but share only the word *on*. Word counts can't see
that *cat ≈ kitten* or *mat ≈ rug*. We need vectors that capture **meaning**.

## Measuring closeness: cosine similarity

Think of each vector as an arrow. Two texts are similar if their arrows **point the same way**,
regardless of length (a long and a short document about cats should still match):

$$\cos(a, b) = \frac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert}$$

`1` = same direction, `0` = unrelated (perpendicular), `-1` = opposite.

### ✏️ Exercise 2.1 - implement cosine similarity

Use `np.dot` and `np.linalg.norm`. Return a Python `float`.

```{code-cell} ipython3
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # YOUR CODE HERE
    raise NotImplementedError
```

```{code-cell} ipython3
from raglab.checks import check_cosine_similarity
check_cosine_similarity(cosine_similarity)
```

## Real embeddings

An **embedding model** is a neural network trained on millions of sentence pairs so that texts
with similar meaning get similar vectors. We use `BAAI/bge-small-en-v1.5`: small (33M
parameters), fast on a CPU, and good for English retrieval. Every text becomes **384 numbers**.

```{code-cell} ipython3
from raglab.embeddings import get_embedder, cosine_similarity as cos_matrix

embedder = get_embedder()
vectors = embedder.embed_passages(sentences, use_cache=False)
print(vectors.shape)                     # 3 sentences x 384 dimensions
print(np.round(vectors[0][:8], 3), "...")
print(np.round(cos_matrix(vectors, vectors), 2))
```

Now *cat on a mat* and *kitten on a rug* are close, and the stock market is far from both.

## Queries vs. passages

Questions and answers look different ("How do I...?" vs. "Return the..."). BGE models were
trained with an instruction in front of **queries only**; `embed_query` adds it for you:

```{code-cell} ipython3
print(repr(embedder.query_prefix))
passages = ["Return the median (middle value) of numeric data.",
            "Delay execution for a given number of seconds.",
            "Recursively delete a directory tree."]
p = embedder.embed_passages(passages, use_cache=False)
for question in ["what's the middle number of a list?", "how to pause my program", "remove a folder and everything in it"]:
    scores = p @ embedder.embed_query(question)   # vectors are normalised: dot product == cosine
    print(f"{question!r:45} -> {passages[int(scores.argmax())]!r}  ({scores.max():.2f})")
```

None of those questions share important words with the right answer - that's the superpower of
**semantic search**.

## Seeing the space

384 dimensions can't be drawn, but we can squash them to 2 with PCA to get a feel for it:

```{code-cell} ipython3
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from raglab import load_documents, chunk_documents

docs = load_documents("python_docs")
picked = ["py.json", "py.csv", "py.pickle", "py.asyncio.tasks", "py.threading", "py.math", "py.statistics"]
chunks = [c for c in chunk_documents(docs, "section") if c.doc_id in picked]
xy = PCA(n_components=2).fit_transform(embedder.embed_passages([c.text for c in chunks]))

plt.figure(figsize=(7, 5))
for doc_id in picked:
    idx = [i for i, c in enumerate(chunks) if c.doc_id == doc_id]
    plt.scatter(xy[idx, 0], xy[idx, 1], label=doc_id, s=18)
plt.legend(fontsize=8); plt.title("Sections of 7 documents, embedded and projected to 2D"); plt.show()
```

Related topics (`json`/`csv`/`pickle` = serialization, `math`/`statistics` = numbers) tend to cluster.

### ✏️ Exercise 2.2 - find the weakness

Embeddings are not magic. Write three queries in the cell below and look at the best match for each:

1. an exact identifier, e.g. `"shutil.copytree"` or an error name;
2. a question whose answer is a specific **number**;
3. a question that uses a **negation** ("which function does NOT ...").

Which ones fail? Keep your notes - lesson 4 (keyword search) fixes some of them.

```{code-cell} ipython3
from raglab.retrieval import VectorRetriever
from raglab.display import hits_table
retriever = VectorRetriever(chunk_documents(docs, "recursive"))
hits_table(retriever.search("shutil.copytree", k=3))   # try your own queries here
```

## Key takeaways

- An embedding is a vector that captures meaning; similar meaning → small angle → high cosine.
- Normalised vectors let you compute cosine with a single dot product (fast!).
- Use the model's query instruction - mixing it up silently hurts quality.
- Embeddings are weak on exact codes, rare names and numbers.

## Going further

- *Simple local RAG* computes embeddings and dot products by hand: `references/mrdbourke_simple-local-rag/00-simple-local-rag.ipynb`
- *pguso/rag-from-scratch* explains embeddings with minimal code (JavaScript): `references/pguso_rag-from-scratch/`
- Choosing a model: the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) ranks embedding models by task.
