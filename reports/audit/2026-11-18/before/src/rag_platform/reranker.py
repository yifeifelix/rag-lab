"""Cross-feature interaction reranker and diversity-aware parent selector."""

from __future__ import annotations

import re
from typing import Sequence

from .retrieval import RetrievedCandidate, tokenize_technical

_IDENT_RE = re.compile(r"\b[A-Z]{2,}[A-Z0-9_.\-]{1,}\b|\b\d+(?:\.\d+)?%?\b")


def _bigrams(tokens: Sequence[str]) -> set[tuple[str, str]]:
    if len(tokens) < 2:
        return set()
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


class CrossFeatureReranker:
    """Fine-grained query-passage cross-interaction reranker.

    Scores retrieved candidates on:
    1. Dense cosine similarity
    2. Normalized BM25 lexical density
    3. Non-stopword query term coverage in passage & parent context
    4. Ordered bigram phrase overlap
    5. Exact identifier / numeric threshold / compliance-code match bonus
    """

    def __init__(self) -> None:
        self._chunk_cache: dict[
            str,
            tuple[
                str,
                frozenset[str],
                frozenset[str],
                frozenset[tuple[str, str]],
                frozenset[str],
                frozenset[str],
            ],
        ] = {}

    def _get_chunk_features(
        self, chunk: object
    ) -> tuple[
        str,
        frozenset[str],
        frozenset[str],
        frozenset[tuple[str, str]],
        frozenset[str],
        frozenset[str],
    ]:
        cid = getattr(chunk, "chunk_id")
        cached = self._chunk_cache.get(cid)
        if cached is not None:
            return cached
        title = getattr(chunk, "title")
        text = getattr(chunk, "text")
        parent_text = getattr(chunk, "parent_text")
        passage_full = f"{title}\n{text}\n{parent_text}"
        p_tokens = tokenize_technical(passage_full, remove_stopwords=True)
        p_token_set = frozenset(p_tokens)
        p_roots = frozenset(t[:6] for t in p_token_set if len(t) >= 6)
        p_bigrams = frozenset(_bigrams(p_tokens))
        title_tokens = frozenset(tokenize_technical(title, remove_stopwords=True))
        t_roots = frozenset(t[:6] for t in title_tokens if len(t) >= 6)
        entry = (passage_full, p_token_set, p_roots, p_bigrams, title_tokens, t_roots)
        self._chunk_cache[cid] = entry
        return entry

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedCandidate],
        top_k: int = 5,
        max_per_doc: int = 2,
    ) -> list[RetrievedCandidate]:
        if not candidates:
            return []

        q_tokens = tokenize_technical(query, remove_stopwords=True)
        q_token_set = set(q_tokens)
        q_bigrams = _bigrams(q_tokens)
        q_idents = set(_IDENT_RE.findall(query))

        max_bm25 = max((c.bm25_score for c in candidates), default=1.0)
        if max_bm25 <= 0:
            max_bm25 = 1.0
        max_rrf = max((c.rrf_score for c in candidates), default=1.0)
        if max_rrf <= 0:
            max_rrf = 1.0

        scored: list[RetrievedCandidate] = []
        for cand in candidates:
            passage_full, p_token_set, p_roots, p_bigrams, title_tokens, t_roots = (
                self._get_chunk_features(cand.chunk)
            )

            # 1. Token coverage ratio (exact + morphological root match) & title alignment
            if q_token_set:
                overlap_count = sum(
                    1
                    for qt in q_token_set
                    if qt in p_token_set or (len(qt) >= 6 and qt[:6] in p_roots)
                )
                token_coverage = overlap_count / len(q_token_set)
                title_overlap = sum(
                    1
                    for qt in q_token_set
                    if qt in title_tokens or (len(qt) >= 6 and qt[:6] in t_roots)
                )
                title_coverage = title_overlap / len(q_token_set)
            else:
                overlap_count = 0
                token_coverage = 0.0
                title_coverage = 0.0

            # 2. Bigram alignment
            if q_bigrams:
                bigram_overlap = len(q_bigrams & p_bigrams) / len(q_bigrams)
            else:
                bigram_overlap = 0.0

            # 3. Exact code / numeric identifier bonus
            ident_bonus = 0.0
            if q_idents:
                hits = sum(1 for ident in q_idents if ident in passage_full)
                ident_bonus = hits / len(q_idents)

            # 4. Normalized RRF, BM25 & Dense signals
            rrf_rel = cand.rrf_score / max_rrf
            bm25_rel = cand.bm25_score / max_bm25
            bm25_abs = min(1.0, cand.bm25_score / 18.0)
            dense_abs = max(0.0, cand.dense_score)

            # Combine first-stage fusion signal + fine cross-passage features
            rerank_val = (
                0.24 * rrf_rel
                + 0.16 * dense_abs
                + 0.16 * bm25_rel
                + 0.22 * token_coverage
                + 0.10 * title_coverage
                + 0.06 * bigram_overlap
                + 0.06 * ident_bonus
            )
            # Absolute calibrated confidence retains grounding check against OOV collapse
            grounding_factor = min(1.0, overlap_count / 3.0)
            abs_confidence = (
                0.35 * dense_abs
                + 0.25 * bm25_abs
                + 0.25 * token_coverage
                + 0.15 * max(bigram_overlap, ident_bonus)
            ) * grounding_factor

            cand.rerank_score = float(rerank_val)
            cand.confidence_score = float(min(cand.confidence_score, abs_confidence) if cand.confidence_score > 0 else abs_confidence)
            scored.append(cand)

        scored.sort(key=lambda c: c.rerank_score, reverse=True)
        return deduplicate_and_select(scored, top_k=top_k, max_per_doc=max_per_doc)


def deduplicate_and_select(
    candidates: Sequence[RetrievedCandidate],
    top_k: int = 5,
    max_per_doc: int = 2,
) -> list[RetrievedCandidate]:
    """Deduplicate identical parent contexts and preserve multi-doc diversity in top-K."""
    selected: list[RetrievedCandidate] = []
    seen_parents: set[tuple[str, str]] = set()
    doc_counts: dict[str, int] = {}

    for cand in candidates:
        parent_key = (cand.chunk.doc_id, cand.chunk.parent_text[:180])
        if parent_key in seen_parents:
            continue
        cnt = doc_counts.get(cand.chunk.doc_id, 0)
        if cnt >= max_per_doc:
            continue
        seen_parents.add(parent_key)
        doc_counts[cand.chunk.doc_id] = cnt + 1
        selected.append(cand)
        if len(selected) >= top_k:
            return selected

    # Backfill if diversity cap filtered too aggressively
    if len(selected) < top_k:
        selected_ids = {c.chunk.chunk_id for c in selected}
        for cand in candidates:
            if cand.chunk.chunk_id in selected_ids:
                continue
            parent_key = (cand.chunk.doc_id, cand.chunk.parent_text[:180])
            if parent_key in seen_parents:
                continue
            seen_parents.add(parent_key)
            selected.append(cand)
            if len(selected) >= top_k:
                break

    return selected
