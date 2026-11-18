"""Finding the chunks that answer a question.

Three retrievers, all with the same interface ``search(query, k) -> list[Hit]``:

* ``VectorRetriever`` - dense / semantic search with embeddings
* ``BM25Retriever``   - classic keyword search
* ``HybridRetriever`` - both, merged with Reciprocal Rank Fusion (RRF)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Protocol, Sequence

import numpy as np

from .chunking import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int  # 1 = best

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id


class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[Hit]: ...


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` largest scores, best first."""
    k = min(k, len(scores))
    if k <= 0:
        return np.array([], dtype=int)
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part])]


# --- Dense retrieval ------------------------------------------------------

class VectorRetriever:
    def __init__(self, chunks: Sequence[Chunk], embedder=None) -> None:
        from .embeddings import get_embedder

        self.chunks = list(chunks)
        self.embedder = embedder or get_embedder()
        self.vectors = self.embedder.embed_passages([c.text for c in self.chunks])

    def search(self, query: str, k: int = 5) -> list[Hit]:
        q = self.embedder.embed_query(query)
        scores = self.vectors @ q  # vectors are normalised, so this IS cosine similarity
        return [Hit(self.chunks[i], float(scores[i]), r) for r, i in enumerate(top_k_indices(scores, k), 1)]


# --- Keyword retrieval ----------------------------------------------------

STOPWORDS = frozenset("""a an and are as at be by can do does for from has have how i in is it its
of on or that the this to was what when where which who why will with you your""".split())
TOKEN_RE = re.compile(r"[a-z0-9_]+(?:\.[a-z0-9_]+)*")


def tokenize(text: str) -> list[str]:
    """Lower-case words; keeps dotted names like ``json.dumps`` together AND
    emits their parts, so both "json.dumps" and "dumps" match."""
    tokens = []
    for match in TOKEN_RE.findall(text.lower()):
        parts = match.split(".")
        if len(parts) > 1:
            tokens.append(match)
        tokens.extend(p for p in parts if p and p not in STOPWORDS)
    return tokens


class BM25Retriever:
    """Okapi BM25.

    score(q, d) = sum over query terms t of
        idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * len(d) / avg_len))
    """

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.doc_tokens = [Counter(tokenize(f"{c.title} {c.text}")) for c in self.chunks]
        self.doc_len = np.array([sum(t.values()) for t in self.doc_tokens], dtype=np.float32)
        self.avg_len = float(self.doc_len.mean()) if len(self.doc_len) else 0.0
        df = Counter(term for tokens in self.doc_tokens for term in tokens)
        n = len(self.chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tokens in enumerate(self.doc_tokens):
                tf = tokens.get(term, 0)
                if tf:
                    norm = 1 - self.b + self.b * self.doc_len[i] / self.avg_len
                    scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return scores

    def search(self, query: str, k: int = 5) -> list[Hit]:
        scores = self.score(query)
        idx = [i for i in top_k_indices(scores, k) if scores[i] > 0]
        return [Hit(self.chunks[i], float(scores[i]), r) for r, i in enumerate(idx, 1)]


# --- Hybrid retrieval -----------------------------------------------------

def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> dict[str, float]:
    """RRF: every list votes 1 / (k + rank) for each item it contains.

    Only ranks are used, so scores on different scales (BM25 vs cosine)
    can be combined fairly. Returns item -> fused score.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRetriever:
    def __init__(self, retrievers: Sequence[Retriever], rrf_k: int = 60, candidates: int = 30) -> None:
        self.retrievers = list(retrievers)
        self.rrf_k = rrf_k
        self.candidates = candidates

    def search(self, query: str, k: int = 5) -> list[Hit]:
        by_id: dict[str, Chunk] = {}
        rankings = []
        for retriever in self.retrievers:
            hits = retriever.search(query, self.candidates)
            rankings.append([h.chunk.chunk_id for h in hits])
            by_id.update({h.chunk.chunk_id: h.chunk for h in hits})
        fused = reciprocal_rank_fusion(rankings, self.rrf_k)
        best = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [Hit(by_id[cid], score, r) for r, (cid, score) in enumerate(best, 1)]
