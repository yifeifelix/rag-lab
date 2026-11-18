"""Re-ranking retrieved candidates with a cross-encoder.

A *bi-encoder* (the embedding model) encodes query and passage separately, so
passage vectors can be pre-computed - fast, but the two texts never "see"
each other. A *cross-encoder* reads query and passage together and outputs a
relevance score - much more accurate, but too slow to run on a whole corpus.
So we retrieve ~20-50 candidates cheaply, then re-rank only those.
"""

from __future__ import annotations

import os
from typing import Sequence

from .retrieval import Hit

DEFAULT_RERANKER = os.getenv("RAGLAB_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
_MODELS: dict[str, object] = {}


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER) -> None:
        from sentence_transformers import CrossEncoder

        if model_name not in _MODELS:
            _MODELS[model_name] = CrossEncoder(model_name, device="cpu")
        self.model = _MODELS[model_name]

    def rerank(self, query: str, hits: Sequence[Hit], k: int = 5) -> list[Hit]:
        if not hits:
            return []
        scores = self.model.predict([(query, h.chunk.text) for h in hits])
        order = sorted(range(len(hits)), key=lambda i: float(scores[i]), reverse=True)[:k]
        return [Hit(hits[i].chunk, float(scores[i]), r) for r, i in enumerate(order, 1)]


class RerankingRetriever:
    """Wrap any retriever: fetch ``candidates`` hits, keep the best ``k`` after re-ranking."""

    def __init__(self, retriever, reranker: CrossEncoderReranker | None = None, candidates: int = 30) -> None:
        self.retriever = retriever
        self.reranker = reranker or CrossEncoderReranker()
        self.candidates = candidates

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return self.reranker.rerank(query, self.retriever.search(query, self.candidates), k)
