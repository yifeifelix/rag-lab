"""Teaching implementations: metadata headers, rule expansion, co-occurrence graph,
heuristic retrieval routing, extractive compression and weighted RRF.
These are inspired by research methods, not official framework integrations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Sequence

from .chunking import Chunk
from .retrieval import RetrievedCandidate, tokenize_technical

# Technical identifiers preserve case; ordinary lower-case prose is not an entity.
_ENTITY_CODE_RE = re.compile(r"\b[A-Z]{2,}[A-Z0-9_.\-]*\b|\b[A-Za-z]+(?:[-_.][A-Za-z0-9]+)+\b")

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


class ContextualHeaderEnricher:
    """Prepends static metadata; this does not generate Anthropic-style chunk explanations."""

    @staticmethod
    def enrich(chunks: Sequence[Chunk]) -> list[Chunk]:
        enriched: list[Chunk] = []
        for c in chunks:
            header = f"[Doc: {c.doc_id} | Domain: {c.domain} | Title: {c.title}]"
            if header not in c.text:
                enriched_text = f"{header}\n{c.text}"
            else:
                enriched_text = c.text
            enriched.append(
                Chunk(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    title=c.title,
                    domain=c.domain,
                    text=enriched_text,
                    parent_text=c.parent_text,
                    metadata={**c.metadata, "contextual_header": True},
                )
            )
        return enriched


class RuleBasedQueryExpander:
    """Explicit operator-supplied dictionary; no built-in benchmark trigger table."""

    def __init__(self, rules=None):
        self.rules = dict(rules or {})

    def expand(self, query: str) -> tuple[str, list[str]]:
        """Return `(expanded_query, added_terms)` without altering out-of-scope queries."""
        lower = query.lower()
        tokens = set(tokenize_technical(query, remove_stopwords=True))
        added: list[str] = []

        # Match both token hits and hyphenated substrings like '8-bit'
        for trigger, expansion in self.rules.items():
            if trigger in tokens or (("-" in trigger or " " in trigger) and trigger in lower):
                for term in expansion.split():
                    if term not in tokens and term not in added:
                        added.append(term)

        if not added:
            return query, []
        expanded = f"{query} {' '.join(added)}"
        return expanded, added


class EntityCooccurrenceGraph:
    """Bounded one-hop technical identifier co-occurrence graph; not LightRAG."""

    def __init__(self, chunks: Sequence[Chunk], max_entities=32, max_neighbors=32,
                 propagation_weight=0.28) -> None:
        self.propagation_weight = propagation_weight
        self.chunks = list(chunks)
        self.entity_to_chunks: dict[str, set[int]] = {}
        self.chunk_to_entities: list[set[str]] = []
        self.entity_idf: dict[str, float] = {}
        self.neighbors: dict[str, dict[str, float]] = {}

        n_chunks = len(self.chunks)
        if n_chunks == 0:
            return

        for idx, c in enumerate(self.chunks):
            combined = f"{c.doc_id} {c.title} {c.text} {c.parent_text}"
            ents = set(sorted(self._extract_entities(combined))[:max_entities])
            self.chunk_to_entities.append(ents)
            for e in ents:
                self.entity_to_chunks.setdefault(e, set()).add(idx)

        for ent, c_ids in self.entity_to_chunks.items():
            self.entity_idf[ent] = math.log(1.0 + (n_chunks + 1.0) / (len(c_ids) + 0.5))

        # Build 1-hop co-occurrence edges within chunks
        for ents in self.chunk_to_entities:
            ent_list = sorted(ents)
            for i in range(len(ent_list)):
                ei = ent_list[i]
                nbrs = self.neighbors.setdefault(ei, {})
                for j in range(len(ent_list)):
                    if i == j:
                        continue
                    ej = ent_list[j]
                    nbrs[ej] = nbrs.get(ej, 0.0) + 1.0

        # Keep a bounded adjacency list even when identifiers occur in many chunks.
        self.neighbors = {ent: dict(sorted(nbrs.items(), key=lambda item: (-item[1], item[0]))[:max_neighbors])
                          for ent, nbrs in self.neighbors.items()}
        # Normalize edge weights per entity
        for ent, nbrs in self.neighbors.items():
            max_w = max(nbrs.values(), default=1.0)
            for nj in nbrs:
                nbrs[nj] = nbrs[nj] / max_w

    @staticmethod
    def _extract_entities(text: str) -> set[str]:
        ents: set[str] = set()
        for m in _ENTITY_CODE_RE.finditer(text):
            val = m.group(0).lower().strip()
            if len(val) >= 2:
                ents.add(val)
        return ents

    def search(
        self,
        query: str,
        top_k: int = 36,
        domain_filter: str | None = None,
    ) -> list[tuple[int, float]]:
        if not self.chunks:
            return []

        q_ents = self._extract_entities(query)
        if not q_ents:
            return []

        # Direct entity activation (1.0) + 1-hop graph neighbor propagation (0.28)
        activated: dict[str, float] = {}
        for qe in q_ents:
            if qe in self.entity_to_chunks:
                activated[qe] = max(activated.get(qe, 0.0), 1.0)
                for nbr, edge_w in self.neighbors.get(qe, {}).items():
                    prop = self.propagation_weight * edge_w
                    if prop > activated.get(nbr, 0.0):
                        activated[nbr] = prop

        if not activated:
            return []

        scores: dict[int, float] = {}
        for ent, act_weight in activated.items():
            idf = self.entity_idf.get(ent, 1.0)
            for c_idx in self.entity_to_chunks.get(ent, ()):
                if domain_filter and self.chunks[c_idx].domain != domain_filter:
                    continue
                scores[c_idx] = scores.get(c_idx, 0.0) + act_weight * idf

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


@dataclass
class RetrievalAssessment:
    """Decision record produced by the Corrective RAG evaluator."""

    status: str  # CORRECT | AMBIGUOUS | INCORRECT
    top_confidence: float
    in_vocab_ratio: float
    trigger_correction: bool


class HeuristicRetrievalRouter:
    """Corrective RAG (CRAG) confidence evaluator & self-correction router.

    Classifies initial retrieval into:
    - INCORRECT: OOV or confidence below refusal threshold -> abstain cleanly.
    - AMBIGUOUS: Mediocre confidence or low term coverage -> trigger query expansion & graph search.
    - CORRECT: High-confidence retrieval -> proceed directly to synthesis.
    """

    def __init__(
        self,
        min_confidence_threshold: float = 0.22,
        ambiguous_upper_threshold: float = 0.48,
        vocab_threshold: float = 0.35,
    ) -> None:
        self.vocab_threshold = vocab_threshold
        self.min_confidence_threshold = min_confidence_threshold
        self.ambiguous_upper_threshold = ambiguous_upper_threshold

    def evaluate(
        self,
        original_query: str,
        candidates: Sequence[RetrievedCandidate],
        vocab: dict[str, int],
    ) -> RetrievalAssessment:
        q_tokens = set(tokenize_technical(original_query, remove_stopwords=True))
        in_vocab_ratio = (
            sum(1 for t in q_tokens if t in vocab) / len(q_tokens) if q_tokens else 0.0
        )
        top_conf = max((c.confidence_score for c in candidates), default=0.0)

        if not candidates or top_conf < self.min_confidence_threshold or in_vocab_ratio < self.vocab_threshold:
            return RetrievalAssessment(
                status="INCORRECT",
                top_confidence=round(top_conf, 4),
                in_vocab_ratio=round(in_vocab_ratio, 4),
                trigger_correction=False,
            )

        if top_conf < self.ambiguous_upper_threshold:
            return RetrievalAssessment(
                status="AMBIGUOUS",
                top_confidence=round(top_conf, 4),
                in_vocab_ratio=round(in_vocab_ratio, 4),
                trigger_correction=True,
            )

        return RetrievalAssessment(
            status="CORRECT",
            top_confidence=round(top_conf, 4),
            in_vocab_ratio=round(in_vocab_ratio, 4),
            trigger_correction=False,
        )


class ContextualCompressor:
    """Prunes irrelevant sentences from retrieved parent contexts to reduce LLM tokens."""

    def compress(
        self,
        query: str,
        parent_text: str,
        max_sentences: int = 4,
    ) -> tuple[str, float]:
        """Return `(compressed_text, compression_ratio)` where ratio is characters retained, not measured token savings."""
        raw_len = max(1, len(parent_text))
        lines = [ln.strip() for ln in _SENTENCE_RE.split(parent_text) if ln.strip()]
        if len(lines) <= max_sentences:
            return parent_text, 1.0

        q_tokens = set(tokenize_technical(query, remove_stopwords=True))
        scored: list[tuple[float, int, str]] = []
        for idx, ln in enumerate(lines):
            # Always keep top markdown title/table header context lightly prioritized
            ln_tokens = set(tokenize_technical(ln, remove_stopwords=True))
            overlap = len(q_tokens & ln_tokens) / max(1, len(q_tokens))
            bonus = 0.25 if (idx == 0 or ln.startswith("|") or ln.startswith("#")) else 0.0
            scored.append((overlap + bonus, idx, ln))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected_indices = sorted(idx for _, idx, _ in scored[:max_sentences])
        # Retain qualifications/negation even if the sentence budget is exceeded.
        constraints = {idx for idx, line in enumerate(lines)
                       if re.search(r'\b(?:not|only|unless|except|must|never|if|before|after|current|archived)\b', line, re.I)}
        selected_indices = sorted(set(selected_indices) | constraints)
        compressed = "\n".join(lines[i] for i in selected_indices)
        ratio = round(len(compressed) / raw_len, 4)
        return compressed, ratio


def fuse_trimodal_rrf(
    dense_hits: Sequence[tuple[int, float]],
    bm25_hits: Sequence[tuple[int, float]],
    graph_hits: Sequence[tuple[int, float]],
    chunks: Sequence[Chunk],
    dense_weight: float = 0.45,
    sparse_weight: float = 0.35,
    graph_weight: float = 0.20,
    rrf_k: int = 60,
    top_k_candidates: int = 36,
) -> list[RetrievedCandidate]:
    """Fuse Dense, BM25, and LightRAG Graph rankings via Tri-Modal Reciprocal Rank Fusion."""
    candidates_map: dict[int, RetrievedCandidate] = {}

    for rank, (idx, score) in enumerate(dense_hits, start=1):
        cand = candidates_map.setdefault(idx, RetrievedCandidate(chunk=chunks[idx]))
        cand.dense_score = score
        cand.dense_rank = rank
        cand.rrf_score += dense_weight * (1.0 / (rrf_k + rank))

    for rank, (idx, score) in enumerate(bm25_hits, start=1):
        cand = candidates_map.setdefault(idx, RetrievedCandidate(chunk=chunks[idx]))
        cand.bm25_score = score
        cand.bm25_rank = rank
        cand.rrf_score += sparse_weight * (1.0 / (rrf_k + rank))

    for rank, (idx, score) in enumerate(graph_hits, start=1):
        cand = candidates_map.setdefault(idx, RetrievedCandidate(chunk=chunks[idx]))
        cand.rrf_score += graph_weight * (1.0 / (rrf_k + rank))

    results = list(candidates_map.values())
    results.sort(key=lambda c: c.rrf_score, reverse=True)
    return results[:top_k_candidates]


# Compatibility aliases for notebooks written before the naming audit.
# Active service/report labels use the accurate implementation names.
QueryExpanderHyDE = RuleBasedQueryExpander
LightRAGEntityGraph = EntityCooccurrenceGraph
CRAGEvaluator = HeuristicRetrievalRouter
CRAGAssessment = RetrievalAssessment
