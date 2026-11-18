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

# 8 · Evaluation: measure, don't guess

**Time:** 60 minutes · **You will:** implement Recall@k and MRR, compare every system you built, and judge answers with an LLM.

## Two questions, two kinds of metrics

1. **Did we retrieve the right evidence?** → retrieval metrics. Cheap, deterministic, need only
   a list of relevant documents per question.
2. **Is the final answer correct and supported?** → answer metrics. Harder; often judged by a
   human or an LLM.

Always evaluate retrieval first: if the evidence isn't retrieved, no prompt can save the answer.

## The question set

```{code-cell} ipython3
import pandas as pd
from raglab import load_qa
qa = load_qa("python_docs")
pd.DataFrame(qa)[["qid", "category", "question", "relevant_doc_ids"]].head(8)
```

```{admonition} Where do question sets come from?
:class: note
These 42 questions were written for this course, and every one stores the exact sentence
(`supporting_spans`) that answers it - a test checks those sentences really exist in the corpus.
For your own projects: collect real user questions, label which documents answer them,
and include questions that *cannot* be answered.
```

## Retrieval metrics

For one question, with retrieved documents in rank order:

- **Recall@k** = (relevant docs in the top k) / (all relevant docs). "Did we find everything we need?"
- **Reciprocal rank** = 1 / rank of the first relevant doc (0 if none). Averaged over questions → **MRR**. "How high is the first good result?"

### ✏️ Exercise 8.1 - Recall@k

```{code-cell} ipython3
def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    # YOUR CODE HERE
    raise NotImplementedError
```

```{code-cell} ipython3
from raglab.checks import check_recall_at_k
check_recall_at_k(recall_at_k)
```

### ✏️ Exercise 8.2 - Reciprocal rank

```{code-cell} ipython3
def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    # YOUR CODE HERE
    raise NotImplementedError
```

```{code-cell} ipython3
from raglab.checks import check_mrr
check_mrr(reciprocal_rank)
```

## The leaderboard: everything you built

```{code-cell} ipython3
from raglab import load_documents, chunk_documents
from raglab.retrieval import BM25Retriever, VectorRetriever, HybridRetriever
from raglab.rerank import RerankingRetriever
from raglab.evaluation import evaluate_retriever, summarize

docs = load_documents("python_docs")
systems = {}
for chunking, kwargs in [("fixed", dict(size=500, overlap=100)), ("recursive", dict(max_chars=800))]:
    chunks = chunk_documents(docs, chunking, **kwargs)
    bm25, vec = BM25Retriever(chunks), VectorRetriever(chunks)
    systems[f"{chunking} | bm25"] = bm25
    systems[f"{chunking} | vector"] = vec
    systems[f"{chunking} | hybrid"] = HybridRetriever([bm25, vec])
systems["recursive | hybrid + rerank"] = RerankingRetriever(systems["recursive | hybrid"])

results = pd.concat([evaluate_retriever(r, qa, k=5, name=n) for n, r in systems.items()])
summarize(results).sort_values("mrr", ascending=False)
```

### Is the difference real?

With 40 questions, one lucky question moves MRR by 0.025. Before believing "A beats B", check how
many questions actually changed:

```{code-cell} ipython3
wide = results.pivot_table(index="qid", columns="system", values="mrr")
a, b = "recursive | hybrid + rerank", "recursive | hybrid"
print(f"{a} better on {(wide[a] > wide[b]).sum()} questions, worse on {(wide[a] < wide[b]).sum()}, "
      f"tied on {(wide[a] == wide[b]).sum()}")
```

```{admonition} Dev vs. test - the most common evaluation mistake
:class: warning
If you tune anything (chunk size, weights, prompts) by looking at these questions, your scores on
them become optimistic. Split your questions: tune on a **dev** set, report once on a **test** set
you never looked at while tuning.
```

## Failure analysis

Numbers tell you *how much*; failures tell you *why*. Group misses by category:

```{code-cell} ipython3
best = results[results.system == "recursive | hybrid + rerank"]
print(best.groupby("category")[["recall@5", "mrr"]].mean().round(2))
best[best["recall@5"] < 1][["qid", "category", "question", "retrieved"]]
```

## Answer quality with an LLM judge

Retrieval can be perfect while the answer is still wrong. An **LLM-as-judge** checks whether every
claim in the answer is supported by the sources (*faithfulness*). Judges are imperfect: spot-check
their verdicts by hand.

```{code-cell} ipython3
from raglab import llm
from raglab.rag import SimpleRAG, format_context
from raglab.evaluation import judge_faithfulness

if llm.llm_available():
    rag = SimpleRAG(systems["recursive | hybrid + rerank"], k=4)
    rows = []
    for item in [q for q in qa if q["relevant_doc_ids"]][:8]:
        out = rag.ask(item["question"])
        verdict = judge_faithfulness(out.answer, format_context(out.hits))
        rows.append({"question": item["question"], "answer": out.answer[:120], "judge": verdict["verdict"],
                     "cited correct doc": bool(set(out.cited_doc_ids) & set(item["relevant_doc_ids"]))})
    display(pd.DataFrame(rows))
else:
    print(llm.explain_setup())
```

### ✏️ Exercise 8.3 - write your own test questions

Read two documents in `data/python_docs/documents.jsonl` you haven't looked at yet. Write 5 new
questions in the cell below (including 1 multi-hop and 1 unanswerable), each with
`relevant_doc_ids`. Run the best system on them. Is it as good as on the course questions? If not,
why might the course set have been "too easy"?

```{code-cell} ipython3
my_qa = [
    # {"qid": "ME-1", "question": "...", "relevant_doc_ids": ["py.textwrap"], "category": "paraphrase"},
]
if my_qa:
    display(summarize(evaluate_retriever(systems["recursive | hybrid + rerank"], my_qa, 5, "mine")))
```

## Key takeaways

- Evaluate retrieval (Recall@k, MRR) before generation.
- Small question sets → small differences are noise; count wins/losses per question.
- Keep a test set you never tune on.
- Read failures by category; that's where the next improvement comes from.

## Going further

- KalyanKS rag-zero-to-hero-guide, *RAG Evaluation* section: `references/KalyanKS-NLP_rag-zero-to-hero-guide/RAG Evaluation/`
- all-in-rag (Chinese), evaluation chapter: `references/datawhalechina_all-in-rag/docs/`
- Frameworks to explore next: RAGAS, DeepEval, TruLens (all implement LLM-judged faithfulness and relevancy).
