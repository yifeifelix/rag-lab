"""Local RAG orchestration with bounded compute, evidence consistency and explicit routing."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field, replace
import os
import hashlib
import json
import logging
import threading
import re
import time
from typing import Any, AsyncIterator, Sequence

import httpx
import numpy as np

from .awesome_rag import (
    ContextualCompressor,
    CRAGEvaluator,
    EntityCooccurrenceGraph,
    ContextualHeaderEnricher,
    RuleBasedQueryExpander,
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
from .reranker import build_reranker, deduplicate_and_select
from .compute import run_cpu
from .security import redact_pii, detects_injection, strip_known_citations, citation_ids
from .observability import event
from .resilience import (
    AsyncConcurrencyGate,
    CircuitBreaker,
    ConcurrencyLimitExceeded,
    TokenBucketRateLimiter,
)
from .retrieval import HybridRRFRetriever, RetrievedCandidate, tokenize_technical

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
    compression_stats: dict[str, float] = field(default_factory=dict)
    retrieval_confidence: float = 0.0
    confidence_kind: str = 'uncalibrated_relevance'
    refusal_reason: str = ''
    unknown_citations: list[str] = field(default_factory=list)
    request_id: str = '' 


# Backwards-compatible imports for old notebooks.
_redact_pii = redact_pii
_detect_prompt_injection = detects_injection


def compute_lexical_grounding(answer: str, evidence: Sequence[ChunkEvidence], refused: bool) -> float | None:
    """Token coverage proxy only. Does not measure factual entailment or correctness."""
    if refused:
        return None
    if not evidence or not answer.strip():
        return 0.0

    # Strip inline citation tags `[DOC-ID]` before token overlap calculation
    clean_answer = strip_known_citations(answer, {ev.doc_id for ev in evidence if hasattr(ev, "doc_id")})
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
            raw = line.strip()
            if raw.startswith('#') or raw.startswith('[Doc:'):
                continue
            if raw.startswith(f'[{doc_id}]') and '(Domain:' in raw:
                continue
            if re.match(r'^(?:Status|Effective(?: version| date)?):', raw, re.I) and not q_tokens & {'status','version','date'}:
                continue
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

    # A simple lookup should not acquire unrelated facts merely to fill a quota.
    multi_part = bool(re.search(r'\b(?:both|compare|between|and|steps|procedure)\b', query, re.I)) or len(q_codes) > 1
    limit = 4 if multi_part else 1
    best_score = scored_sentences[0][0] if scored_sentences else 0.0
    chosen: list[tuple[str, str]] = []
    used_sent: set[str] = set()
    distinct_docs_covered: set[str] = set()

    for score, _, doc_id, sent in scored_sentences:
        if len(chosen) >= limit:
            break
        if score < best_score * 0.75:
            continue
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
    """Single-process learning workbench; capacity must be measured separately."""

    def __init__(
        self,
        config: RAGConfig | None = None,
        documents: Sequence[dict[str, Any]] | None = None,
        previous=None,
    ) -> None:
        self.config = replace(config) if config else RAGConfig()
        self._compute_lock = threading.RLock()
        self._graph_index = None
        self._remote_state = previous._remote_state if previous else {'client': None}
        self.calibrator = None

        if documents is None:
            from .bootstrap import initialize_workspace
            initialize_workspace(self.config)
            self.documents = load_corpus_documents(self.config.corpus_path)
        else:
            self.documents = list(documents)

        self.document_by_id = {document['doc_id']: document for document in self.documents}
        self.document_status = {}
        for doc in self.documents:
            status = re.search(r'(?im)^\s*Status:\s*(CURRENT|ARCHIVED)\b', doc['text'])
            self.document_status[doc['doc_id']] = str(doc.get('status') or (status[1] if status else '')).lower()
        if len(self.document_by_id) != len(self.documents):
            raise ValueError('Duplicate document IDs in corpus')
        # Persist only JSON/NumPy data; snapshots are keyed by corpus + index settings.
        self.corpus_revision = fingerprint(self.documents, self.config)
        index_path = self.config.index_dir / f'{self.corpus_revision}.npz'
        fixed = self.config.embedding_backend == 'sentence_transformers'
        reusable = (fixed and previous is not None and previous.config.embedding_backend == self.config.embedding_backend
                    and previous.config.embedding_model == self.config.embedding_model
                    and (not self.config.embedding_revision or previous.config.embedding_revision == self.config.embedding_revision))
        self.embedder = previous.embedder if reusable else build_embedder(self.config)
        if reusable:
            self._compute_lock = previous._compute_lock
        if fixed and not self.config.embedding_revision:
            self.config.embedding_revision = self.embedder.revision
            self.corpus_revision = fingerprint(self.documents, self.config)
            index_path = self.config.index_dir / f'{self.corpus_revision}.npz'
        self.update_stats = {'encoded_chunks': 0, 'reused_chunks': 0}
        self.index_loaded_from_disk = False
        vectors = None
        if self.config.vector_backend == 'memory' and self.config.persist_index and index_path.exists():
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
            chunk_fields = ('chunking_strategy', 'fixed_chunk_size', 'fixed_overlap', 'recursive_max_chars',
                            'parent_max_chars', 'child_max_chars', 'enable_contextual_headers')
            if reusable and all(getattr(self.config, key) == getattr(previous.config, key) for key in chunk_fields) and not self.config.enable_contextual_headers:
                previous_docs = previous.document_by_id
                old_chunks = {}
                for chunk in previous.chunks:
                    old_chunks.setdefault(chunk.doc_id, []).append(chunk)
                self.chunks = []
                for document in self.documents:
                    if previous_docs.get(document['doc_id']) == document:
                        reused = old_chunks.get(document['doc_id'], [])
                        self.chunks.extend(reused)
                        self.update_stats['reused_chunks'] += len(reused)
                    else:
                        self.chunks.extend(chunker.chunk_documents([document]))
            else:
                self.chunks = chunker.chunk_documents(self.documents)
            if self.config.enable_contextual_headers:
                self.chunks = ContextualHeaderEnricher.enrich(self.chunks)
            self.embedder.fit([c.text for c in self.chunks] + [d["text"] for d in self.documents])

        if (self.config.enable_contextual_headers and not self.index_loaded_from_disk
                and any(not chunk.metadata.get('contextual_header') for chunk in self.chunks)):
            self.chunks = ContextualHeaderEnricher.enrich(self.chunks)
            self.embedder.fit([chunk.text for chunk in self.chunks] + [doc['text'] for doc in self.documents])
        if fixed:
            from .vector_store import encode_cached
            with self._compute_lock:
                vectors, self.update_stats['encoded_chunks'] = encode_cached(self.chunks, self.embedder, self.config)
        sparse_index, token_sets = None, {}
        if fixed:
            from .vector_store import IncrementalBM25Index
            old_sparse = previous.retriever.bm25_index if reusable else None
            sparse_index = IncrementalBM25Index(self.chunks, previous=old_sparse)
            if reusable:
                old_by_id = {chunk.chunk_id: chunk for chunk in previous.chunks}
                token_sets = {chunk.chunk_id: previous.retriever._chunk_token_sets[chunk.chunk_id]
                              for chunk in self.chunks if old_by_id.get(chunk.chunk_id) == chunk}
        self.retriever = HybridRRFRetriever(
            chunks=self.chunks,
            embedder=self.embedder,
            rrf_k=self.config.rrf_k,
            dense_weight=self.config.dense_weight,
            sparse_weight=self.config.sparse_weight,
            embeddings=vectors,
            config=self.config,
            sparse_index=sparse_index,
            token_sets=token_sets,
        )
        if self.config.vector_backend == 'qdrant':
            from .vector_store import QdrantDenseIndex
            self.retriever.dense_index = QdrantDenseIndex(self.chunks, self.embedder, vectors, self.config)
        if self.config.vector_backend == 'memory' and self.config.persist_index and not self.index_loaded_from_disk:
            save_index(index_path, self.chunks, self.embedder, self.retriever.dense_index.embeddings)
        reranker_fields = ('reranker_backend', 'reranker_model', 'reranker_revision', 'feature_rank_weights',
                           'feature_confidence_weights', 'bm25_scale', 'grounding_min_terms')
        reuse_reranker = reusable and all(getattr(self.config, key) == getattr(previous.config, key) for key in reranker_fields)
        self.reranker = previous.reranker if reuse_reranker else build_reranker(self.config)
        if self.config.reranker_backend == 'cross_encoder' and not self.config.reranker_revision:
            self.config.reranker_revision = self.reranker.revision
        rules = {}
        if self.config.expansion_rules_path:
            content = json.loads(self.config.expansion_rules_path.read_text())
            if content.get('role') != 'domain_dictionary' or not content.get('provenance'):
                raise ValueError('Query dictionary needs an explicit role and provenance')
            rules = content['rules']
        self.query_expander = RuleBasedQueryExpander(rules)
        self.crag = CRAGEvaluator(min_confidence_threshold=self.config.min_confidence_threshold,
                                  ambiguous_upper_threshold=self.config.crag_ambiguous_threshold,
                                  vocab_threshold=self.config.crag_vocab_threshold)
        self.compressor = ContextualCompressor()
        if self.config.calibration_path:
            from .study import CalibrationMismatch, ConfidenceCalibration
            try:
                self.calibrator = ConfidenceCalibration.load(self.config.calibration_path, self.config,
                                                              self.corpus_revision)
            except (CalibrationMismatch, OSError):
                event('calibration_stale', level=logging.WARNING, corpus_revision=self.corpus_revision)

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

    @property
    def _http_client(self):
        return self._remote_state['client']

    @_http_client.setter
    def _http_client(self, value):
        self._remote_state['client'] = value

    @property
    def graph_index(self):
        with self._compute_lock:
            if self._graph_index is None:
                self._graph_index = EntityCooccurrenceGraph(self.chunks)
            return self._graph_index

    def execution_signature(self):
        settings = asdict(self.config)
        for key in ('admin_token', 'tester_token', 'reports_dir'):
            settings.pop(key, None)
        settings['query_rules'] = self.query_expander.rules
        settings['openai_model'] = os.getenv('OPENAI_MODEL', '')
        settings['gemini_model'] = os.getenv('GEMINI_MODEL', '')
        return hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()

    def retrieve_context_sync(self, query, q_vec=None, retrieval_mode='hybrid_rrf', use_reranker=True,
                              top_k=5, domain_filter=None, weights=None):
        """Shared retrieval path for serving, grid search and calibration."""
        with self._compute_lock:
            started = time.perf_counter()
            q_vec = q_vec if q_vec is not None else self.embedder.encode_single(query)
            count = max(self.config.top_k_candidates, top_k * 3)
            candidates = self.retriever.retrieve(query, retrieval_mode, count, domain_filter, q_vec)
            initial_confidence = max((c.confidence_score for c in candidates), default=0.0)
            status, rank_query, terms = 'DISABLED', query, []
            if self.config.enable_crag:
                assessment = self.crag.evaluate(query, candidates, self.retriever.bm25_index.vocab)
                status = assessment.status
                if assessment.trigger_correction and self.config.enable_query_expansion:
                    rank_query, terms = self.query_expander.expand(query)
            elif self.config.enable_query_expansion:
                rank_query, terms = self.query_expander.expand(query)
            dw, sw, gw, rk = weights or (self.config.dense_weight, self.config.sparse_weight,
                                        self.config.graph_weight, self.config.rrf_k)
            if retrieval_mode == 'hybrid_rrf' and (self.config.enable_graph_rag or terms or weights):
                vector = q_vec if rank_query == query else self.embedder.encode_single(rank_query)
                dense = self.retriever.dense_index.search_vector(vector, count, domain_filter)
                sparse = self.retriever.bm25_index.search(rank_query, count, domain_filter)
                graph = self.graph_index.search(rank_query, count, domain_filter) if self.config.enable_graph_rag and gw else []
                candidates = fuse_trimodal_rrf(dense, sparse, graph, self.chunks, dw, sw,
                                               gw if graph else 0.0, rk, count)
                self.retriever.score_candidates(query, candidates)
            historical = bool(re.search(r'\b(?:historical|archived|superseded|previous version)\b', query, re.I))
            # Respect explicit source lifecycle metadata; do not infer archive status
            # from an incidental occurrence of the word in a document body.
            candidates = [c for c in candidates if not detects_injection(c.chunk.parent_text)
                          and (historical or self.document_status.get(c.chunk.doc_id) != 'archived')]
            timings = {'retrieval': round((time.perf_counter() - started) * 1000, 3)}
            started = time.perf_counter()
            selected = (self.reranker.rerank(query, candidates, top_k=top_k, max_per_doc=2)
                        if use_reranker else deduplicate_and_select(candidates, top_k, 2))
            timings['rerank'] = round((time.perf_counter() - started) * 1000, 3)
            # Answerability refers to selected evidence, never the initial maximum.
            confidence = max((candidate.confidence_score for candidate in selected), default=0.0)
            return selected, status, terms, confidence, initial_confidence, timings

    def prepare_evidence_sync(self, query, selected):
        with self._compute_lock:
            evidence, effective, ratios = [], [], []
            for candidate in selected:
                text = candidate.chunk.parent_text
                if self.config.enable_contextual_compression:
                    text, ratio = self.compressor.compress(query, text)
                    ratios.append(ratio)
                if not text.strip():
                    continue
                chunk = replace(candidate.chunk, parent_text=text, text=text)
                effective.append(replace(candidate, chunk=chunk))
                document = self.document_by_id[chunk.doc_id]
                evidence.append(ChunkEvidence(chunk.chunk_id, chunk.doc_id, chunk.title, chunk.domain,
                    text[:260], text, candidate.dense_score, candidate.bm25_score, candidate.rrf_score,
                    candidate.rerank_score, candidate.confidence_score,
                    str(document.get('source', '')), str(document.get('version', ''))))
            stats = {'characters_retained_ratio': round(sum(ratios) / len(ratios), 4)} if ratios else {}
            return effective, evidence, stats

    async def _call_remote_llm(self, query, evidence):
        from .generation import generate_remote
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=15.0, trust_env=False,
                limits=httpx.Limits(max_connections=8, max_keepalive_connections=4))
        return await generate_remote(self._http_client, self.config, query, evidence)

    async def aclose(self):
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
        dense = self.retriever.dense_index
        if hasattr(dense, 'close'):
            dense.close()

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

                selected, crag_status, expanded_terms, max_conf, initial_conf, timings = await run_cpu(
                    self.retrieve_context_sync, sanitized_query, q_vec, retrieval_mode, use_reranker,
                    top_k, domain_filter, workers=self.config.cpu_workers)
                stage_timings.update(timings)
                t_gen0 = time.perf_counter()
                if self.config.simulated_llm_latency_ms > 0:
                    await asyncio.sleep(self.config.simulated_llm_latency_ms / 1000.0)

                selected, evidence, compression_stats = await run_cpu(
                    self.prepare_evidence_sync, sanitized_query, selected, workers=self.config.cpu_workers)
                confidence_kind = 'uncalibrated_relevance'
                threshold = self.config.min_confidence_threshold
                calibration_matches_request = (
                    retrieval_mode == self.config.retrieval_mode
                    and use_reranker == self.config.use_reranker
                    and top_k == self.config.top_k_final and domain_filter is None)
                if self.calibrator and calibration_matches_request:
                    max_conf = self.calibrator.predict(max_conf)
                    threshold = self.calibrator.threshold
                    confidence_kind = 'dev_calibrated_answerability'
                refusal_reason = ('retrieval_router_incorrect' if crag_status == 'INCORRECT'
                                  else 'insufficient_evidence' if not selected or max_conf < threshold else '')
                refused = False
                degraded_mode = False
                token_usage: dict[str, int] = {}
                if refusal_reason:
                    refused = True
                    self.refusal_count += 1
                    answer = (
                        "I cannot answer this reliably because the enterprise knowledge base "
                        "contains insufficient grounded evidence (topic not covered or confidence "
                        f"score={max_conf:.2f}, threshold={threshold:.2f}, reason={refusal_reason})."
                    )
                    citations: list[str] = []
                else:
                    if self.config.llm_mode.lower() in ("openai", "gemini") and self.circuit_breaker.allow_upstream():
                        try:
                            answer, token_usage = await self._call_remote_llm(sanitized_query, evidence)
                            self.circuit_breaker.record_success()
                            citations = [cid for cid in dict.fromkeys(ev.doc_id for ev in evidence)
                                         if f'[{cid}]' in answer]
                        except Exception as exc:
                            event('generation_fallback', level=logging.WARNING, provider=self.config.llm_mode,
                                  error_type=type(exc).__name__, corpus_revision=self.corpus_revision)
                            self.circuit_breaker.record_failure()
                            degraded_mode = True
                            answer, citations = await run_cpu(_synthesize_grounded_extractive, sanitized_query, selected, workers=self.config.cpu_workers)
                    else:
                        if self.config.llm_mode.lower() in ("openai", "gemini"):
                            degraded_mode = True
                        answer, citations = await run_cpu(_synthesize_grounded_extractive, sanitized_query, selected, workers=self.config.cpu_workers)

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
                    compression_stats=compression_stats,
                    retrieval_confidence=round(initial_conf, 4),
                    confidence_kind=confidence_kind,
                    refusal_reason=refusal_reason,
                    unknown_citations=[cid for cid in citation_ids(answer) if cid not in {ev.doc_id for ev in evidence}],
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

        signature = self.execution_signature()
        cache_key = signature + make_cache_key(sanitized, mode, rerank_enabled, k_final, domain_filter)
        # Technical identifiers and numbers must agree before semantic answer reuse.
        protected = sorted(set(re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)+|\b\d+\b", sanitized.lower())))
        l2_scope = f"{signature}||{self.corpus_revision}||{mode}||{int(rerank_enabled)}||{k_final}||{domain_filter or '*'}||{protected}"

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
        q_vec = await run_cpu(self.embedder.encode_single, sanitized, workers=self.config.cpu_workers)
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
        async for event in self.replay_response(resp):
            yield event

    @staticmethod
    async def replay_response(resp):
        yield {
            "event": "metadata",
            "data": {
                **asdict(resp),
                'answer': '',
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
            "vector_backend": self.config.vector_backend,
            "reranker_backend": self.config.reranker_backend,
            "graph_built": self._graph_index is not None,
            "update_stats": self.update_stats,
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
