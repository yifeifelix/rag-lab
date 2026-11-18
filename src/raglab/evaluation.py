"""Measuring retrieval and answer quality.

Retrieval metrics only need the question set's ``relevant_doc_ids``:

* Recall@k  - what fraction of the needed documents appear in the top k?
* MRR       - 1 / rank of the first relevant document (0 if none in top k)
* nDCG@k    - rewards relevant documents near the top, discounted by log2(rank + 1)
"""

from __future__ import annotations

import math
import time
from typing import Callable, Sequence

import pandas as pd


def unique_doc_ids(hits) -> list[str]:
    """Collapse chunk hits into document IDs, keeping first-seen order."""
    return list(dict.fromkeys(h.doc_id for h in hits))


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if not relevant:
        raise ValueError("recall is undefined for questions without relevant documents")
    return len(set(retrieved[:k]) & set(relevant)) / len(set(relevant))


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    for rank, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    dcg = sum(1 / math.log2(r + 1) for r, d in enumerate(retrieved[:k], 1) if d in relevant)
    ideal = sum(1 / math.log2(r + 1) for r in range(1, min(len(set(relevant)), k) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate_retriever(retriever, qa: Sequence[dict], k: int = 5, name: str = "") -> pd.DataFrame:
    """Run every answerable question and return one row per question."""
    rows = []
    for item in qa:
        if not item["relevant_doc_ids"]:
            continue
        start = time.perf_counter()
        hits = retriever.search(item["question"], k * 3)  # extra hits: chunks collapse into docs
        docs = unique_doc_ids(hits)[:k]
        rows.append({
            "system": name, "qid": item.get("qid", ""), "category": item.get("category", ""),
            "question": item["question"], "retrieved": docs,
            f"recall@{k}": recall_at_k(docs, item["relevant_doc_ids"], k),
            "mrr": reciprocal_rank(docs, item["relevant_doc_ids"]),
            f"ndcg@{k}": ndcg_at_k(docs, item["relevant_doc_ids"], k),
            "ms": (time.perf_counter() - start) * 1000,
        })
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in results.columns if c.startswith(("recall@", "ndcg@")) or c in ("mrr", "ms")]
    return results.groupby("system", sort=False)[metrics].mean().round(3)


JUDGE_PROMPT = """You are grading a RAG system. Given SOURCES and an ANSWER, decide whether every
factual claim in the ANSWER is supported by the SOURCES.
Reply with one word on the first line - SUPPORTED, PARTIAL or UNSUPPORTED - then one sentence why.

SOURCES:
{sources}

ANSWER:
{answer}"""


def judge_faithfulness(answer: str, sources: str, chat: Callable[[str], str] | None = None) -> dict:
    """LLM-as-judge: is the answer grounded in the retrieved sources?"""
    if chat is None:
        from .llm import chat as default_chat
        chat = default_chat
    reply = chat(JUDGE_PROMPT.format(sources=sources, answer=answer))
    first = reply.strip().split()[0].strip(".:").upper() if reply.strip() else "UNKNOWN"
    return {"verdict": first, "explanation": reply.strip()}
