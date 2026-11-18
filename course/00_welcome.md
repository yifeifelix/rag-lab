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

# 0 · Welcome: what is RAG?

**Time:** 20 minutes · **You will:** see a complete RAG system work end-to-end before we take it apart.

## The problem

A large language model (LLM) like ChatGPT learned from a huge pile of text, but:

1. it **doesn't know your documents** (your company wiki, your notes, last week's policy update);
2. it **can't tell you where an answer came from**;
3. when it doesn't know, it often **makes something up** ("hallucinates").

## The idea: an open-book exam

**Retrieval-Augmented Generation (RAG)** turns a closed-book exam into an open-book one:

```text
          ┌──────────────┐  top-k chunks  ┌───────────────┐
question ─►  RETRIEVER   ├───────────────►  LLM (writer)  ├─► answer + citations
          └──────┬───────┘                └───────────────┘
                 │ searches
          ┌──────▼───────┐
          │ your chunked │  ← built once, offline ("indexing")
          │  documents   │
          └──────────────┘
```

1. **Indexing** (once): split documents into small *chunks* and store them in a searchable index.
2. **Retrieval** (per question): find the few chunks most likely to contain the answer.
3. **Generation** (per question): give those chunks to the LLM and ask it to answer *only* from them, citing its sources.

Almost every "advanced RAG" trick you'll meet is an improvement to one of these three steps.

## How this course works

| Part | Lessons | What you build |
|---|---|---|
| Indexing | 1 Chunking · 2 Embeddings | turn documents into searchable pieces |
| Retrieval | 3 Vector search · 4 BM25 · 5 Hybrid · 6 Re-ranking | find the right evidence |
| Generation | 7 Prompting with sources | grounded answers with citations |
| Quality | 8 Evaluation | measure instead of guessing |
| Advanced | 9 Query transformations · 10 Self-correcting RAG | the techniques from the "awesome RAG" repos |

Every lesson has the same rhythm: **idea → small experiment → exercise (✅ auto-checked) → going further**.

```{admonition} Your files vs. the originals
:class: tip
In JupyterLab you are editing your own copy in `workspace/`. Break things freely!
`scripts/lab reset 03` gives you a fresh copy of lesson 03 (your old one is kept as `.bak`).
Stuck on an exercise? Solutions are in `course/solutions.md` - try first, then compare.
```

## Run your first RAG system

Run each cell with **Shift + Enter**. The first run loads a small embedding model (a few seconds).

```{code-cell} ipython3
from raglab import load_documents, chunk_documents
from raglab.retrieval import VectorRetriever
from raglab.display import hits_table

docs = load_documents("python_docs")      # 45 pages of real Python documentation
print(len(docs), "documents, e.g.", docs[0].title)

chunks = chunk_documents(docs, method="recursive", max_chars=800)
print(len(chunks), "chunks after splitting")

retriever = VectorRetriever(chunks)       # indexing: embed every chunk
```

Now **retrieval** - ask a question in plain English:

```{code-cell} ipython3
hits = retriever.search("How do I turn a Python dict into a JSON string?", k=3)
hits_table(hits)
```

Notice the question never said `dumps` - the embedding model understood the *meaning*.

Finally **generation**. This needs an LLM; if none is configured yet, the cell explains how to set one up (lessons 1-6 don't need it).

```{code-cell} ipython3
from raglab import llm
from raglab.rag import SimpleRAG

if llm.llm_available():
    rag = SimpleRAG(retriever, k=3)
    display(rag.ask("How do I turn a Python dict into a JSON string?"))
else:
    print(llm.explain_setup())
```

## What just happened?

- `chunk_documents` cut 45 documents into a few hundred chunks → **lesson 1**
- `VectorRetriever` turned each chunk into a vector of 384 numbers → **lessons 2-3**
- `search` compared your question's vector with every chunk's vector → **lesson 3**
- `SimpleRAG` pasted the top chunks into a prompt and asked the LLM to cite them → **lesson 7**

## Going further

- *RAG from scratch* (LangChain), videos 1-4 give the same big picture: [GitHub](https://github.com/langchain-ai/rag-from-scratch) · local copy `references/langchain-ai_rag-from-scratch/rag_from_scratch_1_to_4.ipynb`
- *Simple local RAG* builds the whole pipeline in one notebook: [GitHub](https://github.com/mrdbourke/simple-local-rag) · `references/mrdbourke_simple-local-rag/00-simple-local-rag.ipynb`
