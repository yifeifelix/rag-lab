"""Awesome-RAG inspired modules: Contextual Headers, Pseudo-HyDE Query Expansion,
LightRAG Entity Knowledge Graph, Corrective RAG (CRAG), Contextual Compression & AutoRAG Tuner.

References:
- NirDiamant/RAG_Techniques (Contextual Headers, HyDE, CRAG, Contextual Compression)
- HKUDS/LightRAG & DEEP-PolyU/Awesome-GraphRAG (Dual-level entity graph retrieval)
- Marker-Inc-Korea/AutoRAG (Automated pipeline hyperparameter optimization)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Sequence

from .chunking import Chunk
from .retrieval import RetrievedCandidate, tokenize_technical

# Domain conceptual bridge map (Pseudo-HyDE / Query Expansion vocabulary)
_CONCEPTUAL_EXPANSIONS: dict[str, str] = {
    "8-bit": "int8 scalar quantization memory ram recall rescoring",
    "integers": "int8 scalar quantization",
    "compressing": "quantization int8 reduction footprint",
    "memory": "ram footprint hbm",
    "quality": "recall ndcg accuracy precision",
    "vocabulary": "terminology drift shift embedding",
    "shifts": "distribution shift drift psi kl",
    "decay": "degrading drop drift mrr",
    "validate": "evaluation statistical rigor bootstrap paired",
    "statistically": "bootstrap confidence intervals paired hypothesis",
    "noise": "random test sets confidence intervals",
    "viral": "thundering herd singleflight coalescing",
    "duplicate": "singleflight coalescing deduplication future",
    "reworded": "paraphrased semantic cache l2 cosine",
    "paraphrased": "semantic cache l2 cosine similarity",
    "erase": "deletion gdpr cryptographic shredding dek",
    "delete": "erasure dsar-del dek cryptographic shredding",
    "cardholder": "pci-dss pan cvv tokenization",
    "payment": "pci-dss tokenization vault",
    "outage": "circuit breaker failures degraded extractive",
    "500": "circuit breaker degraded extractive",
    "simultaneous": "concurrent concurrency gate burst",
    "capacity": "little law sizing slots workers",
}

_ENTITY_CODE_RE = re.compile(
    r"\b(?:[A-Z]{2,}[A-Z0-9_.\-]{1,}|ERR_[A-Z0-9_]+|SOC2-[A-Z0-9.]+|"
    r"ISO\d+-[A-Z0-9.]+|PCI-DSS-[0-9.]+|HIPAA-[0-9.a-z()]+|SR\s?11-7|"
    r"INT8|FP32|HNSW|BM25|RRF|PSI|KL|LCR|VaR|ES|GDPR|DEK|KMS|mTLS|WORM|"
    r"vLLM|PagedAttention|Singleflight|Little's Law|Parent-Child|InfoNCE|KEDA)\b",
    re.IGNORECASE,
)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


class ContextualHeaderEnricher:
    """Prepends structured document-level metadata headers before indexing (Anthropic Contextual Retrieval)."""

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


class QueryExpanderHyDE:
    """Domain-aware query expansion & pseudo-HyDE bridge for vocabulary mismatches."""

    def expand(self, query: str) -> tuple[str, list[str]]:
        """Return `(expanded_query, added_terms)` without altering out-of-scope queries."""
        lower = query.lower()
        tokens = set(tokenize_technical(query, remove_stopwords=True))
        added: list[str] = []

        # Match both token hits and hyphenated substrings like '8-bit'
        for trigger, expansion in _CONCEPTUAL_EXPANSIONS.items():
            if trigger in tokens or (("-" in trigger or " " in trigger) and trigger in lower):
                for term in expansion.split():
                    if term not in tokens and term not in added:
                        added.append(term)

        if not added:
            return query, []
        expanded = f"{query} {' '.join(added)}"
        return expanded, added


class LightRAGEntityGraph:
    """In-memory dual-level entity & concept knowledge graph inspired by HKUDS/LightRAG.

    Extracts technical entities/codes per chunk, constructs a 1-hop entity co-occurrence
    graph across the enterprise corpus, and retrieves chunks via direct + 1-hop neighbor
    entity activation weighted by inverse chunk frequency.
    """

    def __init__(self, chunks: Sequence[Chunk]) -> None:
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
            ents = self._extract_entities(combined)
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
        # Also include high-signal technical nouns/concepts
        for tok in tokenize_technical(text, remove_stopwords=True):
            if len(tok) >= 4 or any(ch.isdigit() for ch in tok):
                ents.add(tok)
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
                    prop = 0.28 * edge_w
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
class CRAGAssessment:
    """Decision record produced by the Corrective RAG evaluator."""

    status: str  # CORRECT | AMBIGUOUS | INCORRECT
    top_confidence: float
    in_vocab_ratio: float
    trigger_correction: bool


class CRAGEvaluator:
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
    ) -> None:
        self.min_confidence_threshold = min_confidence_threshold
        self.ambiguous_upper_threshold = ambiguous_upper_threshold

    def evaluate(
        self,
        original_query: str,
        candidates: Sequence[RetrievedCandidate],
        vocab: dict[str, int],
    ) -> CRAGAssessment:
        q_tokens = set(tokenize_technical(original_query, remove_stopwords=True))
        in_vocab_ratio = (
            sum(1 for t in q_tokens if t in vocab) / len(q_tokens) if q_tokens else 0.0
        )
        top_conf = max((c.confidence_score for c in candidates), default=0.0)

        if not candidates or top_conf < self.min_confidence_threshold or in_vocab_ratio < 0.35:
            return CRAGAssessment(
                status="INCORRECT",
                top_confidence=round(top_conf, 4),
                in_vocab_ratio=round(in_vocab_ratio, 4),
                trigger_correction=False,
            )

        if top_conf < self.ambiguous_upper_threshold:
            return CRAGAssessment(
                status="AMBIGUOUS",
                top_confidence=round(top_conf, 4),
                in_vocab_ratio=round(in_vocab_ratio, 4),
                trigger_correction=True,
            )

        return CRAGAssessment(
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
        """Return `(compressed_text, compression_ratio)` where lower ratio means fewer tokens."""
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


class AutoRAGTuner:
    """Grid-search hyperparameter optimizer inspired by Marker-Inc-Korea/AutoRAG."""

    @staticmethod
    def tune_rrf_weights(
        retriever: Any,
        graph_index: LightRAGEntityGraph,
        reranker: Any,
        golden_qa: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        """Sweep `(dense_weight, sparse_weight, graph_weight, rrf_k)` and pick best NDCG@5."""
        answerable = [q for q in golden_qa if q.get("relevant_doc_ids")]
        search_space = [
            (0.55, 0.45, 0.00, 60),
            (0.45, 0.35, 0.20, 60),
            (0.40, 0.40, 0.20, 40),
            (0.50, 0.30, 0.20, 60),
            (0.35, 0.45, 0.20, 60),
        ]

        trials: list[dict[str, Any]] = []
        expander = QueryExpanderHyDE()
        for dw, sw, gw, rk in search_space:
            ndcgs: list[float] = []
            mrrs: list[float] = []
            recalls: list[float] = []

            for item in answerable:
                q = item["question"]
                exp_q, _ = expander.expand(q)
                dense_hits = retriever.dense_index.search(exp_q, top_k=36)
                bm25_hits = retriever.bm25_index.search(exp_q, top_k=36)
                graph_hits = graph_index.search(exp_q, top_k=36) if gw > 0 else []

                cands = fuse_trimodal_rrf(
                    dense_hits=dense_hits,
                    bm25_hits=bm25_hits,
                    graph_hits=graph_hits,
                    chunks=retriever.chunks,
                    dense_weight=dw,
                    sparse_weight=sw,
                    graph_weight=gw,
                    rrf_k=rk,
                    top_k_candidates=36,
                )
                ranked = reranker.rerank(exp_q, cands, top_k=5, max_per_doc=2)
                got_docs = list(dict.fromkeys(c.chunk.doc_id for c in ranked))
                rel_set = set(item["relevant_doc_ids"])

                recalls.append(len(rel_set & set(got_docs[:5])) / len(rel_set))
                mrr = 0.0
                for r_idx, doc_id in enumerate(got_docs[:5], start=1):
                    if doc_id in rel_set:
                        mrr = 1.0 / r_idx
                        break
                mrrs.append(mrr)

                dcg = sum(
                    1.0 / math.log2(r_idx + 1.0)
                    for r_idx, doc_id in enumerate(got_docs[:5], start=1)
                    if doc_id in rel_set
                )
                idcg = sum(1.0 / math.log2(r + 1.0) for r in range(1, min(len(rel_set), 5) + 1))
                ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

            trials.append(
                {
                    "dense_weight": dw,
                    "sparse_weight": sw,
                    "graph_weight": gw,
                    "rrf_k": rk,
                    "recall_at_5": round(sum(recalls) / len(recalls), 4),
                    "mrr": round(sum(mrrs) / len(mrrs), 4),
                    "ndcg_at_5": round(sum(ndcgs) / len(ndcgs), 4),
                }
            )

        trials.sort(key=lambda t: (t["ndcg_at_5"], t["mrr"], t["recall_at_5"]), reverse=True)
        return {
            "best_config": trials[0],
            "all_trials": trials,
            "num_queries_evaluated": len(answerable),
        }
