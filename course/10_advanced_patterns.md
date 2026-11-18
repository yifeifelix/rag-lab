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

# 10 · Self-correcting and agentic RAG

**Time:** 60 minutes · **Needs:** an LLM · **You will:** build a RAG loop that checks its own evidence, then map the advanced techniques you can study next.

So far our pipeline is a straight line: retrieve → generate. It never asks *"is this evidence
any good?"* The newest patterns add **decisions** to the loop.

## Corrective RAG (CRAG), simplified

The CRAG paper adds an evaluator between retrieval and generation:

```text
question ─► retrieve ─► grade each chunk ─┬─ some relevant ──► generate from relevant chunks only
                                          └─ none relevant ──► rewrite query, retrieve again
                                                                 └─ still none ──► refuse
```

```{code-cell} ipython3
from raglab import llm, load_documents, chunk_documents
from raglab.retrieval import BM25Retriever, VectorRetriever, HybridRetriever
from raglab.rag import SYSTEM_PROMPT, build_prompt, cited_numbers

chunks = chunk_documents(load_documents("python_docs"), "recursive", max_chars=800)
retriever = HybridRetriever([BM25Retriever(chunks), VectorRetriever(chunks)])
print("LLM ready:", llm.llm_available())
```

### Step 1 · A relevance grader

```{code-cell} ipython3
GRADE_PROMPT = """Does the passage contain information that helps answer the question?
Answer only "yes" or "no".

Question: {question}

Passage:
{passage}"""

def is_relevant(question: str, hit) -> bool:
    reply = llm.chat(GRADE_PROMPT.format(question=question, passage=hit.chunk.text))
    return reply.strip().lower().startswith("yes")

if llm.llm_available():
    q = "How do I remove a directory tree?"
    for hit in retriever.search(q, 4):
        print(is_relevant(q, hit), "|", hit.doc_id, "|", hit.chunk.text[:70].replace("\n", " "))
```

### Step 2 · The corrective loop

```{code-cell} ipython3
REWRITE_PROMPT = """Rewrite this question as a search query for the Python standard library
documentation. Reply with the query only.

Question: {question}"""

def corrective_rag(question: str, k: int = 4, verbose: bool = True) -> str:
    query = question
    for attempt in (1, 2):
        hits = retriever.search(query, k)
        good = [h for h in hits if is_relevant(question, h)]
        if verbose:
            print(f"attempt {attempt}: query={query!r} -> {len(good)}/{len(hits)} relevant")
        if good:
            return llm.chat(build_prompt(question, good), system=SYSTEM_PROMPT)
        query = llm.chat(REWRITE_PROMPT.format(question=question))
    return "I don't know based on the provided sources."

if llm.llm_available():
    for q in ["make the computer wait 2 secs", "What is the capital of Australia?"]:
        print("→", corrective_rag(q), "\n")
```

Cost check: grading 4 chunks = 4 extra LLM calls per question. Cheaper alternatives: grade with the
cross-encoder score from lesson 6, or grade all chunks in one call.

## Agentic RAG: let the model decide

In **agentic RAG**, retrieval becomes a *tool* that the LLM may call - zero, one or many times, with
queries it writes itself. A minimal hand-rolled agent loop:

```{code-cell} ipython3
AGENT_PROMPT = """You can search the Python documentation. On each turn reply with EXACTLY one line:
SEARCH: <query>       to search (you will get the top results), or
ANSWER: <answer>      when you can answer, citing results as [n].
Use at most 3 searches. If the documentation does not help, ANSWER: I don't know.

Question: {question}"""

def agent(question: str, max_steps: int = 4) -> str:
    messages = [{"role": "user", "content": AGENT_PROMPT.format(question=question)}]
    seen = []
    for _ in range(max_steps):
        reply = llm.chat(messages)
        print("🤖", reply.splitlines()[0][:120])
        if reply.startswith("ANSWER:"):
            return reply.removeprefix("ANSWER:").strip()
        query = reply.removeprefix("SEARCH:").strip()
        hits = retriever.search(query, 3)
        seen.extend(hits)
        first = len(seen) - len(hits)
        results = "\n\n".join(f"[{first + i}] {h.chunk.text[:600]}" for i, h in enumerate(hits, 1))
        messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": f"Results:\n{results}"}]
    return "Stopped: step limit reached."

if llm.llm_available():
    print("\nFINAL:", agent("Compress some bytes with gzip and then make them safe to put in JSON as text."))
```

Real frameworks (LangGraph, the OpenAI/Anthropic tool-use APIs) replace the `SEARCH:` text protocol
with structured tool calls, but the loop is the same: **think → act → observe → repeat**.

### ✏️ Exercise 10.1 - CRAG on the leaderboard

Wrap `corrective_rag` so it can be evaluated on the 2 `unanswerable` questions and 10 answerable
ones of your choice. Compare against `SimpleRAG` from lesson 7: does it refuse more correctly? Does
it refuse answerable questions by mistake? How many LLM calls did each use?

```{code-cell} ipython3
# YOUR CODE HERE
```

## Map of advanced techniques

You now understand every building block. Here is where each "awesome RAG" technique plugs in,
with the notebook to study next (local copies in `references/`):

| Stage | Technique | Study notebook |
|---|---|---|
| Indexing | Semantic / proposition chunking | RAG_Techniques `semantic_chunking`, `proposition_chunking` |
| Indexing | Contextual chunk headers, document augmentation | RAG_Techniques `contextual_chunk_headers`, `document_augmentation` |
| Indexing | Hierarchical summaries (RAPTOR) | RAG_Techniques `raptor`, `hierarchical_indices` |
| Indexing | Knowledge graphs (GraphRAG, LightRAG) | RAG_Techniques `graph_rag`, `Microsoft_GraphRag`, `light_rag`; `references/HKUDS_LightRAG/` |
| Indexing | Vectorless, reasoning over a table of contents | `references/VectifyAI_PageIndex/` |
| Retrieval | Fusion, re-ranking, HyDE, Multi-Query | lessons 5, 6, 9 here |
| Retrieval | Context window around chunk, segment extraction | RAG_Techniques `context_enrichment_window_around_chunk`, `relevant_segment_extraction` |
| Post-retrieval | Contextual compression | RAG_Techniques `contextual_compression` |
| Control | Self-RAG, CRAG, adaptive retrieval | RAG_Techniques `self_rag`, `crag`, `adaptive_retrieval`; RAG from scratch 15-18 |
| Control | Agentic RAG with LangGraph | `references/GiovanniPasq_agentic-rag-for-dummies/`; RAG_Techniques `Agentic_RAG` |
| End-to-end | A competition-winning system | `references/IlyaRice_RAG-Challenge-2/` |

**Suggested path:** pick one technique per week; re-implement its core idea with `raglab` on our
corpus; evaluate it with lesson 8's leaderboard. A technique that doesn't move your metrics on
*your* data isn't worth its complexity.

## Key takeaways

- Advanced RAG adds *decisions*: grade evidence, retry, refuse, or search again.
- Every decision is an extra LLM call: watch latency and cost.
- Measure every technique on your own data before adopting it.

## Going further

- CRAG paper: Yan et al., *Corrective Retrieval Augmented Generation* (2024). Self-RAG: Asai et al. (2023).
- RAG from scratch, videos 15-18 (re-ranking, CRAG, Self-RAG, long context): `references/langchain-ai_rag-from-scratch/rag_from_scratch_15_to_18.ipynb`
