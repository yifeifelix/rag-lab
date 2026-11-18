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

# 9 · Query transformations: Multi-Query and HyDE

**Time:** 45 minutes · **Needs:** an LLM · **You will:** let the LLM rewrite questions before retrieval, and measure whether it helps.

Users write short, vague or oddly-worded questions. Instead of searching with the raw question,
we can ask the LLM to **transform** it first. Three classic techniques:

| Technique | Idea | Helps with |
|---|---|---|
| **Rewriting** | rephrase in the documents' vocabulary | vocabulary mismatch (lesson 4) |
| **Multi-Query** | generate several variants, retrieve for each, fuse with RRF | ambiguous questions |
| **HyDE** | write a *hypothetical answer*, search with that | questions that look nothing like answers |

```{code-cell} ipython3
import pandas as pd
from raglab import llm, load_documents, load_qa, chunk_documents
from raglab.retrieval import BM25Retriever, VectorRetriever, HybridRetriever, Hit, reciprocal_rank_fusion
from raglab.evaluation import evaluate_retriever, summarize

chunks = chunk_documents(load_documents("python_docs"), "recursive", max_chars=800)
vec = VectorRetriever(chunks)
hybrid = HybridRetriever([BM25Retriever(chunks), vec])
qa = load_qa("python_docs")
print("LLM ready:", llm.llm_available())
```

## Multi-Query

```{code-cell} ipython3
MULTI_QUERY_PROMPT = """Write 3 different search queries that would find documentation answering
the user's question. Use technical Python terms where helpful. One query per line, no numbering.

Question: {question}"""

def multi_query(question: str) -> list[str]:
    reply = llm.chat(MULTI_QUERY_PROMPT.format(question=question), temperature=0.3)
    return [question] + [line.strip("-• ").strip() for line in reply.splitlines() if line.strip()][:3]

if llm.llm_available():
    print("\n".join(multi_query("how do I make my script stop for a bit")))
```

Retrieve for every variant and fuse the rankings with RRF (lesson 5 again!):

```{code-cell} ipython3
class MultiQueryRetriever:
    def __init__(self, retriever, candidates: int = 20):
        self.retriever, self.candidates = retriever, candidates

    def search(self, query: str, k: int = 5) -> list[Hit]:
        by_id, rankings = {}, []
        for q in multi_query(query):
            hits = self.retriever.search(q, self.candidates)
            rankings.append([h.chunk.chunk_id for h in hits])
            by_id.update({h.chunk.chunk_id: h.chunk for h in hits})
        fused = sorted(reciprocal_rank_fusion(rankings).items(), key=lambda kv: -kv[1])[:k]
        return [Hit(by_id[c], s, r) for r, (c, s) in enumerate(fused, 1)]
```

## HyDE - Hypothetical Document Embeddings

A question ("how to pause?") and its answer ("Delay execution for a given number of seconds")
look different. HyDE asks the LLM to **write a fake answer** in documentation style and embeds
*that*. The fake answer may contain errors - it is only used to *find* real documents, never shown.

```{code-cell} ipython3
HYDE_PROMPT = """Write a short passage (2-3 sentences) from the Python standard library
documentation that answers the question. Write it in the style of a docstring.

Question: {question}"""

class HyDERetriever:
    def __init__(self, vector_retriever):
        self.vec = vector_retriever

    def search(self, query: str, k: int = 5) -> list[Hit]:
        fake = llm.chat(HYDE_PROMPT.format(question=query), temperature=0.0)
        vector = self.vec.embedder.embed_passages([fake], use_cache=False)[0]  # a passage, not a query!
        scores = self.vec.vectors @ vector
        order = scores.argsort()[::-1][:k]
        return [Hit(self.vec.chunks[i], float(scores[i]), r) for r, i in enumerate(order, 1)]

if llm.llm_available():
    print(llm.chat(HYDE_PROMPT.format(question="how do I make my script stop for a bit")))
```

## Measure (this costs LLM calls: ~40 per system)

```{code-cell} ipython3
if llm.llm_available():
    results = pd.concat([
        evaluate_retriever(vec, qa, 5, "vector"),
        evaluate_retriever(hybrid, qa, 5, "hybrid"),
        evaluate_retriever(HyDERetriever(vec), qa, 5, "HyDE (vector)"),
        evaluate_retriever(MultiQueryRetriever(hybrid), qa, 5, "multi-query (hybrid)"),
    ])
    display(summarize(results))
```

Notice the `ms` column: every transformation adds an LLM round-trip **before** retrieval starts.

### ✏️ Exercise 9.1 - query decomposition

Multi-hop questions ("compress with gzip **and then** encode as base64") need evidence from two
documents. Write a `DecomposingRetriever` that asks the LLM to split a question into
independent sub-questions (one per line), retrieves `k // 2` hits for each, and merges them.
Evaluate it on the `multi_hop` questions only:
`[q for q in qa if q["category"] == "multi_hop"]`.

```{code-cell} ipython3
# YOUR CODE HERE
```

## Key takeaways

- Query transformations trade latency and cost for recall.
- HyDE embeds a *fake answer* as a passage; Multi-Query fuses several rankings with RRF.
- They help most where the raw question is vague or worded unlike the documents - measure on
  your own data before adopting them.

## Going further

- RAG_Techniques [query_transformations](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/query_transformations.ipynb), [HyDe](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/HyDe_Hypothetical_Document_Embedding.ipynb), [HyPE](https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/HyPE_Hypothetical_Prompt_Embeddings.ipynb) · `references/NirDiamant_RAG_Techniques/all_rag_techniques/`
- RAG from scratch, videos 5-9 (Multi-Query, RAG-Fusion, decomposition, step-back, HyDE): `references/langchain-ai_rag-from-scratch/rag_from_scratch_5_to_9.ipynb`
- Original HyDE paper: Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels* (2022).
