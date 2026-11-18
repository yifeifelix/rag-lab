"""Vectorized Dense, Okapi BM25, and Hybrid Reciprocal Rank Fusion (RRF) retrievers."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix

from .chunking import Chunk
from .embeddings import SemanticEmbedder

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "both", "by", "can", "describe",
        "did", "do", "does", "explain", "for", "from", "give", "how", "in", "into",
        "is", "it", "list", "me", "of", "on", "or", "state", "tell", "that", "the",
        "their", "them", "these", "this", "to", "under", "was", "what", "when",
        "where", "which", "who", "why", "will", "with", "you", "your",
    }
)

_CODE_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:[-_.][a-zA-Z0-9]+)+")
_WORD_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def tokenize_technical(text: str, remove_stopwords: bool = True) -> list[str]:
    """Tokenize enterprise text preserving technical codes AND emitting sub-tokens."""
    lower = re.sub(r'\[REDACTED_[A-Z]+\]', '', text).lower()
    tokens: list[str] = []

    # Preserve compound technical identifiers (e.g., err_qdrant_5032, soc2-cc6.7, 99.99)
    for match in _CODE_PATTERN.finditer(lower):
        code = match.group(0)
        tokens.append(code)

    # Also extract atomic alphanumeric words
    for match in _WORD_PATTERN.finditer(lower):
        tok = match.group(0)
        if remove_stopwords and tok in _STOPWORDS:
            continue
        if len(tok) >= 2 or tok.isdigit():
            tokens.append(tok)

    return tokens


@dataclass
class RetrievedCandidate:
    """Represents a retrieved chunk with per-stage retrieval and confidence scores."""

    chunk: Chunk
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    confidence_score: float = 0.0
    dense_rank: int | None = None
    bm25_rank: int | None = None


class DenseVectorIndex:
    """In-memory L2-normalized dense cosine similarity index with pre-filtering."""

    def __init__(self, chunks: Sequence[Chunk], embedder: SemanticEmbedder,
                 embeddings: np.ndarray | None = None) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder
        texts = [c.text for c in self.chunks]
        self.embeddings = embeddings if embeddings is not None else embedder.encode(texts)
        self._domain_indices: dict[str, np.ndarray] = {}
        domain_groups: dict[str, list[int]] = {}
        for idx, c in enumerate(self.chunks):
            domain_groups.setdefault(c.domain, []).append(idx)
        for dom, idxs in domain_groups.items():
            self._domain_indices[dom] = np.array(idxs, dtype=np.int64)

    def search_vector(
        self,
        query_vec: np.ndarray,
        top_k: int = 20,
        domain_filter: str | None = None,
    ) -> list[tuple[int, float]]:
        """Search index using precomputed query vector."""
        if len(self.chunks) == 0:
            return []

        if domain_filter:
            allowed = self._domain_indices.get(domain_filter)
            if allowed is None or len(allowed) == 0:
                return []
            sub_scores = self.embeddings[allowed] @ query_vec
            k = min(top_k, len(allowed))
            if k <= 0:
                return []
            top_local = np.argpartition(sub_scores, -k)[-k:]
            top_local = top_local[np.argsort(sub_scores[top_local])[::-1]]
            return [(int(allowed[i]), float(sub_scores[i])) for i in top_local]

        scores = self.embeddings @ query_vec
        k = min(top_k, len(scores))
        if k <= 0:
            return []
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(int(i), float(scores[i])) for i in top_idx]

    def search(
        self,
        query: str,
        top_k: int = 20,
        domain_filter: str | None = None,
    ) -> list[tuple[int, float]]:
        q_vec = self.embedder.encode_single(query)
        return self.search_vector(q_vec, top_k=top_k, domain_filter=domain_filter)


class BM25Index:
    """Vectorized Okapi BM25 sparse lexical index using scipy CSR matrix."""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self._domain_indices: dict[str, np.ndarray] = {}

        n_docs = len(self.chunks)
        if n_docs == 0:
            self.matrix = csr_matrix((0, 0), dtype=np.float32)
            return

        tokenized_docs = [tokenize_technical(c.text) for c in self.chunks]
        lengths = np.array([len(toks) for toks in tokenized_docs], dtype=np.float32)
        avgdl = float(np.mean(lengths)) if np.mean(lengths) > 0 else 1.0

        df: dict[str, int] = {}
        domain_groups: dict[str, list[int]] = {}
        for idx, (c, toks) in enumerate(zip(self.chunks, tokenized_docs)):
            domain_groups.setdefault(c.domain, []).append(idx)
            for t in set(toks):
                df[t] = df.get(t, 0) + 1

        for dom, idxs in domain_groups.items():
            self._domain_indices[dom] = np.array(idxs, dtype=np.int64)

        self.vocab = {term: tid for tid, term in enumerate(sorted(df.keys()))}
        for term, doc_freq in df.items():
            # Lucene-style non-negative BM25 IDF
            self.idf[term] = math.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        for doc_idx, toks in enumerate(tokenized_docs):
            if not toks:
                continue
            tf_counts: dict[str, int] = {}
            for t in toks:
                tf_counts[t] = tf_counts.get(t, 0) + 1
            dl_norm = 1.0 - self.b + self.b * (float(lengths[doc_idx]) / avgdl)
            for term, tf in tf_counts.items():
                tid = self.vocab[term]
                weight = self.idf[term] * (tf * (self.k1 + 1.0)) / (tf + self.k1 * dl_norm)
                rows.append(doc_idx)
                cols.append(tid)
                vals.append(weight)

        self.matrix = csr_matrix(
            (np.array(vals, dtype=np.float32), (np.array(rows), np.array(cols))),
            shape=(n_docs, len(self.vocab)),
            dtype=np.float32,
        )

    def search(
        self,
        query: str,
        top_k: int = 20,
        domain_filter: str | None = None,
    ) -> list[tuple[int, float]]:
        if len(self.chunks) == 0 or not self.vocab:
            return []

        q_tokens = tokenize_technical(query)
        col_ids = [self.vocab[t] for t in set(q_tokens) if t in self.vocab]
        if not col_ids:
            return []

        if domain_filter:
            allowed = self._domain_indices.get(domain_filter)
            if allowed is None or len(allowed) == 0:
                return []
            sub_scores = np.asarray(self.matrix[allowed][:, col_ids].sum(axis=1)).ravel()
            positive_mask = sub_scores > 0
            if not np.any(positive_mask):
                return []
            pos_local = np.where(positive_mask)[0]
            pos_scores = sub_scores[pos_local]
            k = min(top_k, len(pos_local))
            top_rel = np.argpartition(pos_scores, -k)[-k:]
            top_rel = top_rel[np.argsort(pos_scores[top_rel])[::-1]]
            return [(int(allowed[pos_local[i]]), float(pos_scores[i])) for i in top_rel]

        scores = np.asarray(self.matrix[:, col_ids].sum(axis=1)).ravel()
        positive_idx = np.where(scores > 0)[0]
        if len(positive_idx) == 0:
            return []
        pos_scores = scores[positive_idx]
        k = min(top_k, len(positive_idx))
        top_rel = np.argpartition(pos_scores, -k)[-k:]
        top_rel = top_rel[np.argsort(pos_scores[top_rel])[::-1]]
        return [(int(positive_idx[i]), float(pos_scores[i])) for i in top_rel]


class HybridRRFRetriever:
    """Combines Dense and BM25 retrievers using Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: SemanticEmbedder,
        rrf_k: int = 60,
        dense_weight: float = 0.55,
        sparse_weight: float = 0.45,
        embeddings: np.ndarray | None = None,
        config=None,
        sparse_index=None,
        token_sets=None,
    ) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder
        self.dense_index = DenseVectorIndex(self.chunks, embedder, embeddings=embeddings)
        self.bm25_index = sparse_index if sparse_index is not None else BM25Index(self.chunks)
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.config = config
        self._chunk_token_sets: dict[str, frozenset[str]] = {
            c.chunk_id: (token_sets[c.chunk_id] if token_sets and c.chunk_id in token_sets else
                        frozenset(tokenize_technical(f"{c.text} {c.parent_text}", remove_stopwords=True)))
            for c in self.chunks
        }

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid_rrf",
        top_k_candidates: int = 20,
        domain_filter: str | None = None,
        query_vec: np.ndarray | None = None,
    ) -> list[RetrievedCandidate]:
        """Execute dense, bm25, or hybrid_rrf retrieval and return scored candidates."""
        mode_norm = mode.lower().strip()
        q_vec = query_vec if query_vec is not None else self.embedder.encode_single(query)

        dense_hits = (
            self.dense_index.search_vector(q_vec, top_k=top_k_candidates, domain_filter=domain_filter)
            if mode_norm in ("dense", "hybrid_rrf")
            else []
        )
        bm25_hits = (
            self.bm25_index.search(query, top_k=top_k_candidates, domain_filter=domain_filter)
            if mode_norm in ("bm25", "hybrid_rrf")
            else []
        )

        q_tokens = tokenize_technical(query, remove_stopwords=True)
        q_token_set = set(q_tokens)
        in_vocab_ratio = (
            sum(1 for t in q_token_set if t in self.bm25_index.vocab) / len(q_token_set)
            if q_token_set
            else 0.0
        )

        # Normalize max BM25 for calibrated confidence feature
        max_bm25 = max((s for _, s in bm25_hits), default=1.0)
        if max_bm25 <= 0:
            max_bm25 = 1.0

        candidates_map: dict[int, RetrievedCandidate] = {}

        for rank, (idx, score) in enumerate(dense_hits, start=1):
            cand = candidates_map.setdefault(idx, RetrievedCandidate(chunk=self.chunks[idx]))
            cand.dense_score = score
            cand.dense_rank = rank
            if mode_norm == "dense":
                cand.rrf_score = score
            else:
                cand.rrf_score += self.dense_weight * (1.0 / (self.rrf_k + rank))

        for rank, (idx, score) in enumerate(bm25_hits, start=1):
            cand = candidates_map.setdefault(idx, RetrievedCandidate(chunk=self.chunks[idx]))
            cand.bm25_score = score
            cand.bm25_rank = rank
            if mode_norm == "bm25":
                cand.rrf_score = score
            else:
                cand.rrf_score += self.sparse_weight * (1.0 / (self.rrf_k + rank))

        results = list(candidates_map.values())
        self.score_candidates(query, results, mode_norm)
        results.sort(key=lambda c: c.rrf_score, reverse=True)
        return results[:top_k_candidates]

    def score_candidates(self, query, results, mode_norm='hybrid_rrf'):
        q_token_set = set(tokenize_technical(query, remove_stopwords=True))
        in_vocab_ratio = (sum(token in self.bm25_index.vocab for token in q_token_set) / len(q_token_set)
                          if q_token_set else 0.0)
        for cand in results:
            p_tokens = self._chunk_token_sets.get(cand.chunk.chunk_id)
            if p_tokens is None:
                p_tokens = frozenset(
                    tokenize_technical(f"{cand.chunk.text} {cand.chunk.parent_text}", remove_stopwords=True)
                )
            hits = len(q_token_set & p_tokens) if q_token_set else 0
            cov = hits / len(q_token_set) if q_token_set else 0.0
            # Grounding dampener: suppresses OOV single-token cosine collapses
            floor = self.config.query_vocab_floor if self.config else 0.45
            minimum = self.config.grounding_min_terms if self.config else 3
            grounding_factor = min(1.0, hits / max(1, min(minimum, len(q_token_set)))) * min(1.0, in_vocab_ratio / floor)

            bm25_abs = min(1.0, cand.bm25_score / (self.config.bm25_scale if self.config else 18.0))
            dense_abs = max(0.0, cand.dense_score)
            if mode_norm == "dense":
                weights = self.config.dense_confidence_weights if self.config else (0.60, 0.40)
                raw_conf = sum(weight * value for weight, value in zip(weights, (dense_abs, cov)))
            elif mode_norm == "bm25":
                weights = self.config.sparse_confidence_weights if self.config else (0.65, 0.35)
                raw_conf = sum(weight * value for weight, value in zip(weights, (bm25_abs, cov)))
            else:
                weights = self.config.hybrid_confidence_weights if self.config else (0.45, 0.35, 0.20)
                raw_conf = sum(weight * value for weight, value in zip(weights, (dense_abs, bm25_abs, cov)))
            cand.confidence_score = float(raw_conf * grounding_factor)

