"""Single-owner RAG workbench: queries, document lifecycle, explicit experiments."""

import asyncio
from dataclasses import asdict, replace
import hmac
import json
import os
import time
import uuid
import threading
import logging
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator

from .benchmark import run_concurrency_benchmark
from .config import RAGConfig
from .data_gen import load_golden_qa
from .evaluation import run_ablation_experiments
from .knowledge_base import KnowledgeBase, RevisionConflict
from .runtime import configure_runtime
from .compute import run_cpu
from .observability import configure_logging, event
from .security import redact_pii
from .study import RRFGridSearch, calibrate, load_dev

STATIC_DIR = Path(__file__).resolve().parent / 'static'


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    retrieval_mode: str = Field(default='hybrid_rrf', pattern='^(dense|bm25|hybrid_rrf)$')
    use_reranker: bool = True
    use_cache: bool = True
    top_k: int = Field(default=5, ge=1, le=15)
    domain_filter: str | None = None


class BenchmarkRequest(BaseModel):
    total_requests: int = Field(default=2000, ge=100, le=10000)
    simulated_llm_latency_ms: float = Field(default=2.0, ge=0.0, le=50.0)


class DocumentRequest(BaseModel):
    doc_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$')
    title: str = Field(min_length=1, max_length=250)
    domain: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=20, max_length=500_000)
    source: str = Field(default='', max_length=2000)
    version: str = Field(default='1', max_length=80)
    license: str = Field(default='', max_length=250)

    @field_validator('title', 'domain', 'text')
    @classmethod
    def require_content(cls, value):
        if not value.strip():
            raise ValueError('Content must not be blank')
        return value


class FeedbackRequest(BaseModel):
    request_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    rating: int = Field(ge=1, le=5)
    issue: str = Field(pattern=r'^(correct|wrong_answer|bad_citation|wrong_refusal|slow|other)$')
    comment: str = Field(default='', max_length=2000)


def create_app(config: RAGConfig | None = None) -> FastAPI:
    configure_runtime(config.project_root if config else None)
    configure_logging()
    cfg = config or RAGConfig()
    kb = KnowledgeBase(cfg)
    @asynccontextmanager
    async def lifespan(app):
        yield
        await kb.aclose()
    app = FastAPI(title='RAG Learning & Evaluation Workbench', version='1.2.0', lifespan=lifespan)
    issued = OrderedDict()
    feedback_lock = threading.Lock()

    def record_response(request_id, response):
        issued[request_id] = {'request_id': request_id, 'corpus_revision': response.corpus_revision,
            'generation_mode': response.generation_mode, 'refused': response.refused,
            'confidence': response.confidence, 'citations': response.citations}
        while len(issued) > 10000:
            issued.popitem(last=False)

    @app.middleware('http')
    async def request_events(request, call_next):
        request.state.request_id = uuid.uuid4().hex
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            event('request_failed', level=logging.ERROR, request_id=request.state.request_id,
                  error_type=type(exc).__name__)
            response = JSONResponse(status_code=500, content={'detail': 'Internal error; consult the request ID',
                                                             'request_id': request.state.request_id})
        response.headers['X-Request-ID'] = request.state.request_id
        response.headers['Cache-Control'] = 'no-store'
        event('http_request', request_id=request.state.request_id, method=request.method,
              status=response.status_code, role=getattr(request.state, 'role', 'anonymous'),
              duration_ms=round((time.perf_counter()-started)*1000, 3))
        return response
    app.state.config = cfg
    app.state.knowledge_base = kb

    async def authorize(request: Request, authorization: str | None = Header(default=None)):
        provided = authorization or ''
        if cfg.admin_token and hmac.compare_digest(provided, f'Bearer {cfg.admin_token}'):
            request.state.role = 'owner'
        elif cfg.tester_token and hmac.compare_digest(provided, f'Bearer {cfg.tester_token}'):
            request.state.role = 'tester'
        elif not cfg.admin_token and request.client and request.client.host in ('127.0.0.1', '::1', 'testclient'):
            request.state.role = 'owner'
        elif cfg.admin_token or cfg.tester_token:
            raise HTTPException(401, 'Access token required', headers={'WWW-Authenticate': 'Bearer'})
        else:
            raise HTTPException(403, 'Configure access tokens before remote access')
        if request.state.role == 'tester':
            allowed = (request.method == 'GET' and (request.url.path in ('/api/v1/stats', '/api/v1/capabilities')
                        or request.url.path.startswith('/api/v1/documents')))
            allowed |= request.method == 'POST' and request.url.path in ('/api/v1/query', '/api/v1/query/stream', '/api/v1/feedback')
            if not allowed:
                raise HTTPException(403, 'Owner permission required')

    api = APIRouter(prefix='/api/v1', dependencies=[Depends(authorize)])

    async def current():
        return await run_cpu(kb.current, workers=cfg.cpu_workers)

    @app.get('/', include_in_schema=False)
    async def index():
        return FileResponse(STATIC_DIR / 'index.html')

    @app.get('/health')
    async def health():
        pipeline = await current()
        key_name = {'openai': 'OPENAI_API_KEY', 'gemini': 'GEMINI_API_KEY'}.get(cfg.llm_mode)
        ready = not key_name or (bool(os.getenv(key_name))
                                and (cfg.llm_mode != 'gemini' or bool(os.getenv('GEMINI_MODEL'))))
        dense = pipeline.retriever.dense_index
        vector_ready = await run_cpu(dense.ready, workers=cfg.cpu_workers) if hasattr(dense, 'ready') else True
        return {'status': 'healthy' if ready and vector_ready else 'degraded', 'llm_mode': cfg.llm_mode,
                'provider_key_configured': ready, 'embedding_backend': cfg.embedding_backend,
                'vector_store_ready': vector_ready,
                'corpus_documents': len(pipeline.documents), 'indexed_chunks': len(pipeline.chunks),
                'chunking_strategy': cfg.chunking_strategy, 'corpus_revision': pipeline.corpus_revision,
                'index_loaded_from_disk': pipeline.index_loaded_from_disk,
                'vector_backend': cfg.vector_backend, 'reranker_backend': cfg.reranker_backend,
                'calibration_loaded': pipeline.calibrator is not None}

    @api.post('/query')
    async def query_rag(req: QueryRequest, request: Request):
        response = await (await current()).query(**req.model_dump())
        if response.cache_tier == 'rate_limited':
            raise HTTPException(429, 'Service busy; retry with backoff', headers={'Retry-After': '1'})
        response = replace(response, request_id=request.state.request_id)
        record_response(response.request_id, response)
        return asdict(response)

    @api.post('/query/stream')
    async def stream_rag(req: QueryRequest, request: Request):
        pipeline = await current()
        response = await pipeline.query(**req.model_dump())
        if response.cache_tier == 'rate_limited':
            raise HTTPException(429, 'Service busy; retry with backoff', headers={'Retry-After': '1'})
        response = replace(response, request_id=request.state.request_id)
        record_response(response.request_id, response)
        async def events():
            async for event in pipeline.replay_response(response):
                if event['event'] == 'metadata':
                    event['data']['request_id'] = request.state.request_id
                    issued[request.state.request_id] = {'request_id': request.state.request_id,
                        'corpus_revision': pipeline.corpus_revision, 'generation_mode': event['data']['generation_mode'],
                        'confidence': event['data']['confidence'], 'citations': event['data']['citations']}
                    while len(issued) > 10000:
                        issued.popitem(last=False)
                yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"
        return StreamingResponse(events(), media_type='text/event-stream',
                                 headers={'X-Streaming-Mode': 'buffered-response-replay'})

    @api.get('/documents')
    async def list_documents():
        return await asyncio.to_thread(kb.list_documents)

    @api.get('/documents/{doc_id}')
    async def get_document(doc_id: str):
        pipeline = await current()
        document = pipeline.document_by_id.get(doc_id)
        if document:
            return {'document': document, 'corpus_revision': pipeline.corpus_revision}
        raise HTTPException(404, 'Document not found')

    async def mutate(document=None, delete_id=None, expected_revision=None):
        try:
            return await asyncio.to_thread(kb.mutate, document, delete_id, expected_revision)
        except RevisionConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, 'Document not found') from exc
        except ValueError as exc:
            raise HTTPException(422, 'Document could not be indexed; existing corpus retained') from exc

    @api.post('/documents')
    async def upsert_document(req: DocumentRequest, if_match: str | None = Header(default=None)):
        return await mutate(req.model_dump(), expected_revision=if_match)

    @api.put('/documents/{doc_id}')
    async def update_document(doc_id: str, req: DocumentRequest, if_match: str | None = Header(default=None)):
        if doc_id != req.doc_id:
            raise HTTPException(422, 'Path and body document IDs must agree')
        return await mutate(req.model_dump(), expected_revision=if_match)

    @api.delete('/documents/{doc_id}')
    async def delete_document(doc_id: str, if_match: str | None = Header(default=None)):
        return await mutate(delete_id=doc_id, expected_revision=if_match)

    @api.get('/ablation')
    async def get_ablation(refresh: bool = False):
        report = Path(cfg.reports_dir) / 'ablation_results.json'
        if report.exists() and not refresh:
            from .provenance import sha256_file
            payload = json.loads(report.read_text())
            provenance = payload.get('provenance', {})
            payload['report_matches_current_data'] = (
                provenance.get('corpus_sha256') == sha256_file(cfg.corpus_path)
                and provenance.get('qa_sha256') == sha256_file(cfg.golden_qa_path)
                and provenance.get('source_sha256') == {
                    path.name: sha256_file(path) for path in Path(__file__).parent.glob('*.py')})
            return payload
        if not refresh:
            raise HTTPException(404, 'Run an architecture comparison explicitly first (refresh=true)')
        return await asyncio.to_thread(lambda: asyncio.run(run_ablation_experiments(cfg, save_reports=True)))

    @api.get('/benchmark')
    async def get_benchmark():
        report = Path(cfg.reports_dir) / 'benchmark_10k.json'
        if not report.exists():
            raise HTTPException(404, 'Run the synthetic benchmark explicitly first')
        return json.loads(report.read_text())

    @api.post('/benchmark/run')
    async def run_live_benchmark(req: BenchmarkRequest):
        return await asyncio.to_thread(lambda: asyncio.run(run_concurrency_benchmark(
            total_requests=req.total_requests, simulated_llm_latency_ms=req.simulated_llm_latency_ms,
            save_reports=req.total_requests == 10000)))

    @api.post('/rrf/tune')
    @api.post('/autorag/tune', deprecated=True)
    async def tune_autorag():
        try:
            rows = await run_cpu(load_dev, cfg, workers=cfg.cpu_workers)
            return await run_cpu(RRFGridSearch.tune, await current(), rows, workers=cfg.cpu_workers)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @api.post('/calibration/run')
    async def run_calibration():
        try:
            rows = await run_cpu(load_dev, cfg, workers=cfg.cpu_workers)
            result = await calibrate(await current(), rows)
            path = cfg.reports_dir / 'calibration.json'
            from .bootstrap import atomic_text
            await run_cpu(atomic_text, path, json.dumps(result, indent=2) + '\n', workers=cfg.cpu_workers)
            return {'artifact': str(path), 'applied': False, **result}
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @api.get('/capabilities')
    async def capabilities(request: Request):
        return {'role': request.state.role, 'can_edit_documents': request.state.role == 'owner',
                'can_run_experiments': request.state.role == 'owner', 'workspace_scope': 'shared_single_workspace'}

    @api.post('/feedback')
    async def feedback(req: FeedbackRequest):
        record = issued.get(req.request_id)
        if record is None:
            raise HTTPException(404, 'Unknown response ID; run the query again after a restart')
        def save():
            import fcntl
            from datetime import datetime, timezone
            path = cfg.reports_dir / 'user_feedback.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            value = {**record, 'rating': req.rating, 'issue': req.issue,
                     'comment': redact_pii(req.comment)[0], 'time': datetime.now(timezone.utc).isoformat()}
            with feedback_lock, path.open('a', encoding='utf-8') as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                stream.write(json.dumps(value, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
        await run_cpu(save, workers=cfg.cpu_workers)
        return {'saved': True, 'request_id': req.request_id}

    @api.get('/stats')
    async def get_stats():
        pipeline = await current()
        qa = load_golden_qa(cfg.dev_qa_path) if cfg.dev_qa_path.exists() else []
        return {'telemetry': pipeline.get_stats(),
                'domains': sorted({d['domain'] for d in pipeline.documents}),
                'sample_queries': [{'question_id': q['question_id'], 'category': q['category'],
                                    'question': q['question'],
                                    'paraphrase': next(iter(q.get('paraphrase_variants', [])), q['question'])}
                                   for q in qa[:24]]}

    app.include_router(api)
    return app
