"""EnterpriseRAGPipeline: End-to-end async RAG orchestrator built for 10k concurrency."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import os
import re
import time
from typing import Any, AsyncIterator, Sequence

import httpx
import numpy as np

from .awesome_rag import (
    ContextualCompressor,
    CRAGEvaluator,
    LightRAGEntityGraph,
    QueryExpanderHyDE,
    fuse_trimodal_rrf,
)
from .cache import (
    L1ExactCache,
    L2SemanticCache,
    SingleflightCoalescer,
    clone_dataclass_with,
    make_cache_key,
)
from .chunking import Chunk, build_chunker
from .config import RAGConfig
from .data_gen import generate_datasets, load_corpus_documents
from .embeddings import build_embedder
from .index_store import fingerprint, load_index, save_index
from .reranker import CrossFeatureReranker, deduplicate_and_select
from .resilience import (
    AsyncConcurrencyGate,
    CircuitBreaker,
    ConcurrencyLimitExceeded,
    TokenBucketRateLimiter,
)
from .retrieval import HybridRRFRetriever, RetrievedCandidate, tokenize_technical

# Prompt-injection & exfiltration guardrail patterns
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(?:all\s+)?previous\s+instructions",
        r"disregard\s+(?:all\s+)?(?:system|previous)\s+(?:rules|instructions)",
        r"override\s+(?:all\s+)?previous\s+instructions",
        r"reveal\s+(?:your\s+)?(?:hidden\s+)?system\s+prompt",
        r"dump\s+(?:hidden\s+)?system\s+prompt",
        r"exfiltrate\s+(?:secrets|keys|credentials)",
        r"print\s+secret\s+(?:kms|api)\s+(?:credentials|keys)",
    )
]

# PII patterns scrubbed prior to retrieval/logging
_PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[REDACTED_PAN]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]+=*\b", re.IGNORECASE), "Bearer [REDACTED_TOKEN]"),
]

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class ChunkEvidence:
    """Serialized retrieval evidence returned in RAGResponse."""

    chunk_id: str
    doc_id: str
    title: str
    domain: str
    excerpt: str
    parent_text: str
    dense_score: float
    bm25_score: float
    rrf_score: float
    rerank_score: float
    confidence_score: float
    source: str = ''
    version: str = ''


@dataclass
class RAGResponse:
    """Structured enterprise RAG query response with full observability telemetry."""

    query: str
    sanitized_query: str
    answer: str
    citations: list[str]
    evidence: list[ChunkEvidence]
    retrieval_mode: str
    use_reranker: bool
    cache_tier: str  # miss | l1_exact | l2_semantic | singleflight_coalesced | blocked | rate_limited
    l2_similarity: float
    confidence: float
    lexical_grounding_proxy: float | None
    refused: bool
    blocked_by_guardrail: bool
    pii_redacted: bool
    degraded_mode: bool
    latency_ms: float
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    generation_mode: str = 'local_extractive'
    embedding_backend: str = 'local_svd'
    corpus_revision: str = ''
    token_usage: dict[str, int] = field(default_factory=dict)
    estimated_cost_usd: float | None = None
    crag_status: str = 'DISABLED'
    query_expanded_terms: list[str] = field(default_factory=list)


def _redact_pii(text: str) -> tuple[str, bool]:
    redacted = text
    changed = False
    for pattern, replacement in _PII_PATTERNS:
        new_text, count = pattern.subn(replacement, redacted)
        if count > 0:
            changed = True
            redacted = new_text
    return redacted, changed


def _detect_prompt_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def compute_lexical_grounding(answer: str, evidence: Sequence[ChunkEvidence], refused: bool) -> float | None:
    """Token coverage proxy only. Does not measure factual entailment or correctness."""
    if refused:
        return None
    if not evidence or not answer.strip():
        return 0.0

    # Strip inline citation tags `[DOC-ID]` before token overlap calculation
    clean_answer = re.sub(r"\[[A-Z0-9\-]+\]", "", answer)
    ans_tokens = tokenize_technical(clean_answer, remove_stopwords=True)
    if not ans_tokens:
        return 1.0

    ctx_tokens: set[str] = set()
    for ev in evidence:
        ctx_tokens.update(tokenize_technical(ev.parent_text, remove_stopwords=True))
        ctx_tokens.update(tokenize_technical(ev.title, remove_stopwords=True))

    if not ctx_tokens:
        return 0.0
    grounded = sum(1 for t in ans_tokens if t in ctx_tokens)
    return float(min(1.0, grounded / len(ans_tokens)))


def _synthesize_grounded_extractive(
    query: str,
    selected: Sequence[RetrievedCandidate],
) -> tuple[str, list[str]]:
    """Synthesize a multi-doc grounded response with inline `[doc_id]` citations."""
    q_tokens = set(tokenize_technical(query, remove_stopwords=True))
    q_codes = set(re.findall(r"[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)+", query))

    # Collect scored sentences across distinct documents to support multi-hop answers
    scored_sentences: list[tuple[float, int, str, str]] = []
    seen_sentences: set[str] = set()

    for doc_rank, cand in enumerate(selected):
        doc_id = cand.chunk.doc_id
        raw_lines = _SENTENCE_SPLIT_RE.split(cand.chunk.parent_text)
        for line in raw_lines:
            sent = line.strip().lstrip("#").strip()
            if len(sent) < 20 or sent.startswith("| ---"):
                continue
            norm_s = sent.lower()
            if norm_s in seen_sentences:
                continue
            seen_sentences.add(norm_s)

            s_tokens = set(tokenize_technical(sent, remove_stopwords=True))
            overlap = len(q_tokens & s_tokens) / max(1, len(q_tokens))
            code_hits = sum(1 for c in q_codes if c.lower() in norm_s)
            score = (
                2.5 * overlap
                + 1.5 * code_hits
                + 0.4 / (1 + doc_rank)
                + (0.15 if "|" in sent else 0.0)
            )
            scored_sentences.append((score, doc_rank, doc_id, sent))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)

    # Ensure at least one top sentence from each top-2 distinct document for multi-hop breadth
    chosen: list[tuple[str, str]] = []
    used_sent: set[str] = set()
    distinct_docs_covered: set[str] = set()

    for cand in selected[:3]:
        for score, _, doc_id, sent in scored_sentences:
            if doc_id == cand.chunk.doc_id and sent not in used_sent:
                chosen.append((doc_id, sent))
                used_sent.add(sent)
                distinct_docs_covered.add(doc_id)
                break

    for score, _, doc_id, sent in scored_sentences:
        if len(chosen) >= 4:
            break
        if sent not in used_sent:
            chosen.append((doc_id, sent))
            used_sent.add(sent)
            distinct_docs_covered.add(doc_id)

    if not chosen and selected:
        top = selected[0]
        chosen.append((top.chunk.doc_id, top.chunk.text.strip()))

    rendered = " ".join(f"{sent} [{doc_id}]" for doc_id, sent in chosen)
    citations: list[str] = []
    for doc_id, _ in chosen:
        if doc_id not in citations:
            citations.append(doc_id)
    return rendered, citations


class EnterpriseRAGPipeline:
    """Production-architected async RAG pipeline supporting ablation & 10k concurrency."""

    def __init__(
        self,
        config: RAGConfig | None = None,
        documents: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        self.config = config or RAGConfig()

        if documents is None:
            if not self.config.corpus_path.exists() or not self.config.golden_qa_path.exists():
                generate_datasets(self.config, force=False)
            self.documents = load_corpus_documents(self.config.corpus_path)
        else:
            self.documents = list(documents)

        # Persist only JSON/NumPy data; snapshots are keyed by corpus + index settings.
        self.corpus_revision = fingerprint(self.documents, self.config)
        index_path = self.config.index_dir / f'{self.corpus_revision}.npz'
        self.embedder = build_embedder(self.config)
        self.index_loaded_from_disk = False
        vectors = None
        if self.config.persist_index and index_path.exists():
            try:
                self.chunks, vectors = load_index(index_path, self.embedder)
                self.index_loaded_from_disk = True
            except (ValueError, KeyError, OSError):
                # Rebuild a damaged or incompatible disposable snapshot.
                vectors = None
        if self.index_loaded_from_disk:
            pass
        elif not self.documents:
            self.chunks = []
            self.embedder.fit(['empty knowledge index placeholder', 'no documents available'])
        elif self.config.chunking_strategy == "semantic":
            # Pre-fit embedder on document texts so SemanticBreakpointChunker reuses it
            self.embedder.fit([d["text"] for d in self.documents])
            chunker = build_chunker(
                "semantic",
                embedder=self.embedder,
                semantic_breakpoint_percentile=self.config.semantic_breakpoint_percentile,
            )
            self.chunks: list[Chunk] = chunker.chunk_documents(self.documents)
            # Re-fit on both document texts and chunk texts for richer vocab coverage
            self.embedder.fit([c.text for c in self.chunks] + [d["text"] for d in self.documents])
        else:
            chunker = build_chunker(
                self.config.chunking_strategy,
                fixed_chunk_size=self.config.fixed_chunk_size,
                fixed_overlap=self.config.fixed_overlap,
                recursive_max_chars=self.config.recursive_max_chars,
                parent_max_chars=self.config.parent_max_chars,
                child_max_chars=self.config.child_max_chars,
            )
            self.chunks = chunker.chunk_documents(self.documents)
            self.embedder.fit([c.text for c in self.chunks] + [d["text"] for d in self.documents])

        self.retriever = HybridRRFRetriever(
            chunks=self.chunks,
            embedder=self.embedder,
            rrf_k=self.config.rrf_k,
            dense_weight=self.config.dense_weight,
            sparse_weight=self.config.sparse_weight,
            embeddings=vectors,
        )
        if self.config.persist_index and not self.index_loaded_from_disk:
            save_index(index_path, self.chunks, self.embedder, self.retriever.dense_index.embeddings)
        self.reranker = CrossFeatureReranker()
        self.graph_index = LightRAGEntityGraph(self.chunks)
        self.query_expander = QueryExpanderHyDE()
        self.crag = CRAGEvaluator(min_confidence_threshold=self.config.min_confidence_threshold)
        self.compressor = ContextualCompressor()

        # Multi-tier caching & coalescing
        self.l1_cache = L1ExactCache(
            max_size=self.config.l1_max_size,
            ttl_seconds=self.config.l1_ttl_seconds,
        )
        self.l2_cache = L2SemanticCache(
            max_size=self.config.l2_max_size,
            similarity_threshold=self.config.l2_similarity_threshold,
            ttl_seconds=self.config.l2_ttl_seconds,
        )
        self.singleflight = SingleflightCoalescer()

        # Resilience controls
        self.rate_limiter = TokenBucketRateLimiter(
            rps=self.config.rate_limit_rps,
            burst=self.config.rate_limit_burst,
        )
        self.concurrency_gate = AsyncConcurrencyGate(
            max_concurrent=self.config.max_concurrent_pipelines,
            queue_timeout_seconds=self.config.queue_timeout_seconds,
        )
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.circuit_breaker_failure_threshold,
            reset_seconds=self.config.circuit_breaker_reset_seconds,
        )

        # Runtime telemetry counters
        self.total_queries: int = 0
        self.guardrail_blocks: int = 0
        self.refusal_count: int = 0
        self.cache_misses: int = 0

    def reset_runtime_state(self) -> None:
        """Reset caches and telemetry counters cleanly between benchmarks."""
        self.l1_cache.clear()
        self.l2_cache.clear()
        self.singleflight.reset_stats()
        self.rate_limiter.reset_stats()
        self.concurrency_gate.reset_stats()
        self.circuit_breaker.reset()
        self.total_queries = 0
        self.guardrail_blocks = 0
        self.refusal_count = 0
        self.cache_misses = 0

    async def _call_remote_llm(self, query: str, evidence: Sequence[ChunkEvidence]) -> tuple[str, dict[str, int]]:
        """Optional OpenAI / Gemini synthesis when configured; guarded by CircuitBreaker."""
        context_block = "\n\n".join(
            f"[{ev.doc_id}] ({ev.title}):\n{ev.parent_text}" for ev in evidence
        )
        system_prompt = (
            "You are an enterprise RAG assistant. Answer strictly from <retrieved_context> "
            "and cite doc IDs like [CLOUD-ARCH-001]. If insufficient, state that explicitly."
        )
        user_prompt = f"<retrieved_context>\n{context_block}\n</retrieved_context>\n\nQuestion: {query}"

        mode = self.config.llm_mode.lower()
        if mode == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY missing")
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                        "temperature": 0.0,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    },
                )
                resp.raise_for_status()
                payload = resp.json()
                usage = payload.get('usage', {})
                return payload['choices'][0]['message']['content'].strip(), {
                    'input_tokens': usage.get('prompt_tokens', 0),
                    'output_tokens': usage.get('completion_tokens', 0)}

        if mode == "gemini":
            api_key = os.getenv("GEMINI_API_KEY", "")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY missing")
            model = os.getenv('GEMINI_MODEL')
            if not model:
                raise RuntimeError('GEMINI_MODEL must be explicitly configured')
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    params={"key": api_key},
                    json={
                        "contents": [
                            {"parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}
                        ]
                    },
                )
                resp.raise_for_status()
                payload = resp.json()
                usage = payload.get('usageMetadata', {})
                return payload['candidates'][0]['content']['parts'][0]['text'].strip(), {
                    'input_tokens': usage.get('promptTokenCount', 0),
                    'output_tokens': usage.get('candidatesTokenCount', 0)}

        raise ValueError(f"Unsupported llm_mode: {mode}")

    async def _execute_backend_pipeline(
        self,
        sanitized_query: str,
        q_vec: np.ndarray,
        pii_redacted: bool,
        retrieval_mode: str,
        use_reranker: bool,
        top_k: int,
        domain_filter: str | None,
        cache_key: str,
        l2_scope: str,
        use_cache: bool,
    ) -> RAGResponse:
        stage_timings: dict[str, float] = {}
        try:
            async with self.concurrency_gate.acquire() as wait_ms:
                stage_timings["queue_wait"] = round(wait_ms, 3)

                # 1. Retrieval (plus optional Awesome-RAG LightRAG Graph + CRAG + Query Expansion)
                t_ret0 = time.perf_counter()
                top_k_cand = max(self.config.top_k_candidates, top_k * 3)
                candidates = self.retriever.retrieve(
                    query=sanitized_query,
                    mode=retrieval_mode,
                    top_k_candidates=top_k_cand,
                    domain_filter=domain_filter,
                    query_vec=q_vec,
                )
                orig_max_conf = max((c.confidence_score for c in candidates), default=0.0)
                orig_q_tokens = set(tokenize_technical(sanitized_query, remove_stopwords=True))
                orig_in_vocab = (
                    sum(1 for t in orig_q_tokens if t in self.retriever.bm25_index.vocab)
                    / len(orig_q_tokens)
                    if orig_q_tokens
                    else 0.0
                )

                rank_query = sanitized_query
                expanded_terms: list[str] = []
                crag_status = "DISABLED"

                if self.config.enable_crag:
                    assessment = self.crag.evaluate(
                        sanitized_query,
                        candidates,
                        self.retriever.bm25_index.vocab,
                    )
                    crag_status = assessment.status
                    if assessment.trigger_correction or (
                        self.config.enable_query_expansion and assessment.status != "INCORRECT"
                    ):
                        rank_query, expanded_terms = self.query_expander.expand(sanitized_query)
                        if expanded_terms:
                            crag_status = "AMBIGUOUS_CORRECTED"
                elif self.config.enable_query_expansion and orig_max_conf >= self.config.min_confidence_threshold and orig_in_vocab >= 0.35:
                    rank_query, expanded_terms = self.query_expander.expand(sanitized_query)

                should_run_trimodal = (
                    retrieval_mode == "hybrid_rrf"
                    and orig_max_conf >= self.config.min_confidence_threshold
                    and orig_in_vocab >= 0.35
                    and (self.config.enable_graph_rag or bool(expanded_terms))
                )
                if should_run_trimodal:
                    dense_hits = self.retriever.dense_index.search(
                        rank_query, top_k=top_k_cand, domain_filter=domain_filter
                    )
                    bm25_hits = self.retriever.bm25_index.search(
                        rank_query, top_k=top_k_cand, domain_filter=domain_filter
                    )
                    graph_hits = (
                        self.graph_index.search(
                            rank_query, top_k=top_k_cand, domain_filter=domain_filter
                        )
                        if (self.config.enable_graph_rag or self.config.enable_crag)
                        else []
                    )
                    candidates = fuse_trimodal_rrf(
                        dense_hits=dense_hits,
                        bm25_hits=bm25_hits,
                        graph_hits=graph_hits,
                        chunks=self.chunks,
                        dense_weight=0.45 if graph_hits else self.config.dense_weight,
                        sparse_weight=0.35 if graph_hits else self.config.sparse_weight,
                        graph_weight=0.20 if graph_hits else 0.0,
                        rrf_k=self.config.rrf_k,
                        top_k_candidates=top_k_cand,
                    )
                    for c in candidates:
                        c.confidence_score = orig_max_conf

                stage_timings["retrieval"] = round((time.perf_counter() - t_ret0) * 1000.0, 3)

                # 2. Reranking or Diversity-aware selection
                t_rr0 = time.perf_counter()
                if use_reranker:
                    selected = self.reranker.rerank(
                        query=rank_query,
                        candidates=candidates,
                        top_k=top_k,
                        max_per_doc=2,
                    )
                else:
                    selected = deduplicate_and_select(candidates, top_k=top_k, max_per_doc=2)
                stage_timings["rerank"] = round((time.perf_counter() - t_rr0) * 1000.0, 3)

                # 3. Confidence Check & Synthesis
                t_gen0 = time.perf_counter()
                if rank_query != sanitized_query or should_run_trimodal:
                    max_conf = orig_max_conf
                else:
                    max_conf = max((c.confidence_score for c in selected), default=0.0)
                if self.config.simulated_llm_latency_ms > 0:
                    await asyncio.sleep(self.config.simulated_llm_latency_ms / 1000.0)

                ratios: list[float] = []
                evidence = []
                for c in selected:
                    p_text = c.chunk.parent_text
                    if self.config.enable_contextual_compression:
                        p_text, ratio = self.compressor.compress(sanitized_query, p_text)
                        ratios.append(ratio)
                    evidence.append(
                        ChunkEvidence(
                            chunk_id=c.chunk.chunk_id,
                            doc_id=c.chunk.doc_id,
                            title=c.chunk.title,
                            domain=c.chunk.domain,
                            excerpt=c.chunk.text[:260],
                            parent_text=p_text,
                            dense_score=round(c.dense_score, 4),
                            bm25_score=round(c.bm25_score, 4),
                            rrf_score=round(c.rrf_score, 4),
                            rerank_score=round(c.rerank_score, 4),
                            confidence_score=round(c.confidence_score, 4),
                            source=next((str(d.get('source', '')) for d in self.documents
                                         if d['doc_id'] == c.chunk.doc_id), ''),
                            version=next((str(d.get('version', '')) for d in self.documents
                                          if d['doc_id'] == c.chunk.doc_id), ''),
                        )
                    )
                if ratios:
                    stage_timings["compression_ratio"] = round(sum(ratios) / len(ratios), 4)

                refused = False
                degraded_mode = False
                token_usage: dict[str, int] = {}
                if not selected or max_conf < self.config.min_confidence_threshold:
                    refused = True
                    self.refusal_count += 1
                    answer = (
                        "I cannot answer this reliably because the enterprise knowledge base "
                        "contains insufficient grounded evidence (topic not covered or confidence "
                        f"{max_conf:.2f} below threshold {self.config.min_confidence_threshold:.2f})."
                    )
                    citations: list[str] = []
                else:
                    if self.config.llm_mode.lower() in ("openai", "gemini") and self.circuit_breaker.allow_upstream():
                        try:
                            answer, token_usage = await self._call_remote_llm(sanitized_query, evidence)
                            self.circuit_breaker.record_success()
                            citations = [cid for cid in dict.fromkeys(ev.doc_id for ev in evidence)
                                         if f'[{cid}]' in answer]
                        except Exception:
                            self.circuit_breaker.record_failure()
                            degraded_mode = True
                            answer, citations = _synthesize_grounded_extractive(sanitized_query, selected)
                    else:
                        if self.config.llm_mode.lower() in ("openai", "gemini"):
                            degraded_mode = True
                        answer, citations = _synthesize_grounded_extractive(sanitized_query, selected)

                stage_timings["synthesis"] = round((time.perf_counter() - t_gen0) * 1000.0, 3)

                lexical_grounding_proxy = compute_lexical_grounding(answer, evidence, refused=refused)
                generation_mode = ('refusal' if refused else 'extractive_fallback' if degraded_mode
                                   else 'local_extractive' if self.config.llm_mode == 'local'
                                   else self.config.llm_mode)
                # Pricing is supplied by the operator, never inferred from stale price lists.
                estimated_cost = 0.0 if generation_mode in ('refusal', 'local_extractive') else None
                in_rate = os.getenv('RAG_INPUT_USD_PER_MILLION')
                out_rate = os.getenv('RAG_OUTPUT_USD_PER_MILLION')
                if token_usage and in_rate and out_rate:
                    estimated_cost = (token_usage['input_tokens'] * float(in_rate)
                                      + token_usage['output_tokens'] * float(out_rate)) / 1_000_000

                response = RAGResponse(
                    query=sanitized_query,
                    sanitized_query=sanitized_query,
                    answer=answer,
                    citations=citations,
                    evidence=evidence,
                    retrieval_mode=retrieval_mode,
                    use_reranker=use_reranker,
                    cache_tier="miss",
                    l2_similarity=0.0,
                    confidence=round(max_conf, 4),
                    lexical_grounding_proxy=round(lexical_grounding_proxy, 4) if lexical_grounding_proxy is not None else None,
                    refused=refused,
                    blocked_by_guardrail=False,
                    pii_redacted=pii_redacted,
                    degraded_mode=degraded_mode,
                    latency_ms=0.0,
                    stage_timings_ms=stage_timings,
                    generation_mode=generation_mode,
                    embedding_backend=self.config.embedding_backend,
                    corpus_revision=self.corpus_revision,
                    token_usage=token_usage,
                    estimated_cost_usd=estimated_cost,
                    crag_status=crag_status,
                    query_expanded_terms=expanded_terms,
                )

                if use_cache and not degraded_mode:
                    self.l1_cache.set(cache_key, response)
                    # Lexical eligibility is a heuristic, not factual verification.
                    if not refused and lexical_grounding_proxy >= 0.70:
                        self.l2_cache.set(q_vec, l2_scope, response)

                self.cache_misses += 1
                return response

        except ConcurrencyLimitExceeded:
            return RAGResponse(
                query=sanitized_query,
                sanitized_query=sanitized_query,
                answer="Service busy: concurrency limit reached. Please retry with backoff.",
                citations=[],
                evidence=[],
                retrieval_mode=retrieval_mode,
                use_reranker=use_reranker,
                cache_tier="rate_limited",
                l2_similarity=0.0,
                confidence=0.0,
                lexical_grounding_proxy=None,
                generation_mode="refusal",
                embedding_backend=self.config.embedding_backend,
                corpus_revision=self.corpus_revision,
                estimated_cost_usd=0.0,
                refused=True,
                blocked_by_guardrail=False,
                pii_redacted=pii_redacted,
                degraded_mode=True,
                latency_ms=0.0,
                stage_timings_ms=stage_timings,
            )

    async def query(
        self,
        query: str,
        retrieval_mode: str | None = None,
        use_reranker: bool | None = None,
        use_cache: bool | None = None,
        top_k: int | None = None,
        domain_filter: str | None = None,
    ) -> RAGResponse:
        """Execute an end-to-end enterprise RAG query."""
        t0 = time.perf_counter()
        self.total_queries += 1

        mode = (retrieval_mode or self.config.retrieval_mode).lower().strip()
        rerank_enabled = self.config.use_reranker if use_reranker is None else use_reranker
        cache_enabled = self.config.use_cache if use_cache is None else use_cache
        k_final = top_k or self.config.top_k_final

        # Stage 0: Security Guardrails (PII redaction + prompt injection block)
        sanitized, pii_redacted = _redact_pii(query)
        if _detect_prompt_injection(sanitized):
            self.guardrail_blocks += 1
            return RAGResponse(
                query=sanitized,
                sanitized_query=sanitized,
                answer=(
                    "Request blocked by security guardrails (ERR_GUARD_4007): "
                    "potential prompt injection or credential exfiltration attempt detected; cannot comply."
                ),
                citations=[],
                evidence=[],
                retrieval_mode=mode,
                use_reranker=rerank_enabled,
                cache_tier="blocked",
                l2_similarity=0.0,
                confidence=0.0,
                lexical_grounding_proxy=None,
                generation_mode="refusal",
                embedding_backend=self.config.embedding_backend,
                corpus_revision=self.corpus_revision,
                estimated_cost_usd=0.0,
                refused=True,
                blocked_by_guardrail=True,
                pii_redacted=pii_redacted,
                degraded_mode=False,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                stage_timings_ms={"guardrail": round((time.perf_counter() - t0) * 1000.0, 3)},
            )

        # Stage 1: Token Bucket Rate Limiter
        if not self.rate_limiter.allow():
            return RAGResponse(
                query=sanitized,
                sanitized_query=sanitized,
                answer="Rate limit exceeded (ERR_GW_4290). Please retry shortly.",
                citations=[],
                evidence=[],
                retrieval_mode=mode,
                use_reranker=rerank_enabled,
                cache_tier="rate_limited",
                l2_similarity=0.0,
                confidence=0.0,
                lexical_grounding_proxy=None,
                generation_mode="refusal",
                embedding_backend=self.config.embedding_backend,
                corpus_revision=self.corpus_revision,
                estimated_cost_usd=0.0,
                refused=True,
                blocked_by_guardrail=False,
                pii_redacted=pii_redacted,
                degraded_mode=True,
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            )

        cache_key = make_cache_key(sanitized, mode, rerank_enabled, k_final, domain_filter)
        # Technical identifiers and numbers must agree before semantic answer reuse.
        protected = sorted(set(re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)+|\b\d+\b", sanitized.lower())))
        l2_scope = f"{self.corpus_revision}||{mode}||{int(rerank_enabled)}||{k_final}||{domain_filter or '*'}||{protected}"

        # Stage 2: L1 Exact Cache
        if cache_enabled:
            l1_hit = self.l1_cache.get(cache_key)
            if l1_hit is not None:
                return clone_dataclass_with(
                    l1_hit,
                    query=sanitized,
                    sanitized_query=sanitized,
                    cache_tier="l1_exact",
                    token_usage={}, estimated_cost_usd=0.0,
                    l2_similarity=1.0,
                    pii_redacted=pii_redacted,
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                    stage_timings_ms={"l1_cache": round((time.perf_counter() - t0) * 1000.0, 3)},
                )

        # Stage 3: Encode Query Vector & Check L2 Semantic Cache
        t_emb0 = time.perf_counter()
        q_vec = self.embedder.encode_single(sanitized)
        emb_ms = round((time.perf_counter() - t_emb0) * 1000.0, 3)

        if cache_enabled:
            l2_hit = self.l2_cache.get(q_vec, l2_scope)
            if l2_hit is not None:
                cached_resp, sim = l2_hit
                return clone_dataclass_with(
                    cached_resp,
                    query=sanitized,
                    sanitized_query=sanitized,
                    cache_tier="l2_semantic",
                    token_usage={}, estimated_cost_usd=0.0,
                    l2_similarity=round(sim, 4),
                    pii_redacted=pii_redacted,
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                    stage_timings_ms={
                        "embedding": emb_ms,
                        "l2_cache": round((time.perf_counter() - t0) * 1000.0, 3),
                    },
                )

        # Stage 4: Singleflight Coalescing + Backend Pipeline
        async def _leader() -> RAGResponse:
            return await self._execute_backend_pipeline(
                sanitized_query=sanitized,
                q_vec=q_vec,
                pii_redacted=pii_redacted,
                retrieval_mode=mode,
                use_reranker=rerank_enabled,
                top_k=k_final,
                domain_filter=domain_filter,
                cache_key=cache_key,
                l2_scope=l2_scope,
                use_cache=cache_enabled,
            )

        if self.config.enable_singleflight:
            result, was_coalesced = await self.singleflight.execute(cache_key, _leader)
            if was_coalesced:
                return clone_dataclass_with(
                    result,
                    query=sanitized,
                    sanitized_query=sanitized,
                    pii_redacted=pii_redacted,
                    cache_tier="rate_limited" if result.cache_tier == "rate_limited" else "singleflight_coalesced",
                    token_usage={}, estimated_cost_usd=0.0,
                    latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                )
        else:
            result = await _leader()

        timings = dict(result.stage_timings_ms)
        timings["embedding"] = emb_ms
        return clone_dataclass_with(
            result,
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            stage_timings_ms=timings,
        )

    async def stream_query(
        self,
        query: str,
        retrieval_mode: str | None = None,
        use_reranker: bool | None = None,
        use_cache: bool | None = None,
        top_k: int | None = None,
        domain_filter: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream RAG response as structured Server-Sent Event payloads."""
        resp = await self.query(
            query=query,
            retrieval_mode=retrieval_mode,
            use_reranker=use_reranker,
            use_cache=use_cache,
            top_k=top_k,
            domain_filter=domain_filter,
        )
        yield {
            "event": "metadata",
            "data": {
                "citations": resp.citations,
                "cache_tier": resp.cache_tier,
                "confidence": resp.confidence,
                "retrieval_mode": resp.retrieval_mode,
                "use_reranker": resp.use_reranker,
                "evidence_count": len(resp.evidence),
                "generation_mode": resp.generation_mode,
                "streaming_mode": "buffered_response_replay",
                "corpus_revision": resp.corpus_revision,
            },
        }
        tokens = resp.answer.split(" ")
        for i, tok in enumerate(tokens):
            piece = tok if i == 0 else f" {tok}"
            yield {"event": "delta", "data": {"token": piece}}
            await asyncio.sleep(0)
        yield {
            "event": "done",
            "data": {
                "latency_ms": resp.latency_ms,
                "lexical_grounding_proxy": resp.lexical_grounding_proxy,
                "refused": resp.refused,
                "blocked_by_guardrail": resp.blocked_by_guardrail,
            },
        }

    def get_stats(self) -> dict[str, Any]:
        """Return live telemetry snapshot for monitoring dashboards."""
        total = max(1, self.total_queries)
        l1_hits = self.l1_cache.hits
        l2_hits = self.l2_cache.hits
        sf_hits = self.singleflight.coalesced_hits
        return {
            "corpus_documents": len(self.documents),
            "indexed_chunks": len(self.chunks),
            "chunking_strategy": self.config.chunking_strategy,
            "embedding_backend": self.config.embedding_backend,
            "llm_mode": self.config.llm_mode,
            "corpus_revision": self.corpus_revision,
            "index_loaded_from_disk": self.index_loaded_from_disk,
            "embedding_dim": self.embedder.svd.n_components if self.embedder.svd else self.embedder.embedding_dim,
            "total_queries": self.total_queries,
            "l1_cache_size": len(self.l1_cache),
            "l1_hits": l1_hits,
            "l2_cache_size": len(self.l2_cache),
            "l2_hits": l2_hits,
            "singleflight_coalesced_hits": sf_hits,
            "cache_misses": self.cache_misses,
            "effective_cache_hit_rate": round((l1_hits + l2_hits + sf_hits) / total, 4),
            "guardrail_blocks": self.guardrail_blocks,
            "refusal_count": self.refusal_count,
            "peak_concurrency": self.concurrency_gate.peak_concurrency,
            "circuit_breaker_state": self.circuit_breaker.state,
        }
