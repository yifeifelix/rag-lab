"""Single-owner RAG workbench: queries, document lifecycle, explicit experiments."""

import asyncio
from dataclasses import asdict
import hmac
import json
import os
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from .benchmark import run_concurrency_benchmark
from .config import RAGConfig
from .data_gen import load_golden_qa
from .evaluation import run_ablation_experiments
from .knowledge_base import KnowledgeBase, RevisionConflict

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


def create_app(config: RAGConfig | None = None) -> FastAPI:
    cfg = config or RAGConfig()
    kb = KnowledgeBase(cfg)
    app = FastAPI(title='RAG Learning & Evaluation Workbench', version='1.1.0')
    app.state.config = cfg
    app.state.knowledge_base = kb

    async def authorize(request: Request, authorization: str | None = Header(default=None)):
        if cfg.admin_token:
            if not hmac.compare_digest(authorization or '', f'Bearer {cfg.admin_token}'):
                raise HTTPException(401, 'Owner token required', headers={'WWW-Authenticate': 'Bearer'})
        elif request.client is None or request.client.host not in ('127.0.0.1', '::1', 'testclient'):
            raise HTTPException(403, 'Configure RAG_ADMIN_TOKEN for access beyond loopback')

    api = APIRouter(prefix='/api/v1', dependencies=[Depends(authorize)])

    async def current():
        return await asyncio.to_thread(kb.current)

    @app.get('/', include_in_schema=False)
    async def index():
        return FileResponse(STATIC_DIR / 'index.html')

    @app.get('/health')
    async def health():
        pipeline = await current()
        key_name = {'openai': 'OPENAI_API_KEY', 'gemini': 'GEMINI_API_KEY'}.get(cfg.llm_mode)
        ready = not key_name or (bool(os.getenv(key_name))
                                and (cfg.llm_mode != 'gemini' or bool(os.getenv('GEMINI_MODEL'))))
        return {'status': 'healthy' if ready else 'degraded', 'llm_mode': cfg.llm_mode,
                'provider_key_configured': ready, 'embedding_backend': cfg.embedding_backend,
                'corpus_documents': len(pipeline.documents), 'indexed_chunks': len(pipeline.chunks),
                'chunking_strategy': cfg.chunking_strategy, 'corpus_revision': pipeline.corpus_revision,
                'index_loaded_from_disk': pipeline.index_loaded_from_disk}

    @api.post('/query')
    async def query_rag(req: QueryRequest):
        response = await (await current()).query(**req.model_dump())
        if response.cache_tier == 'rate_limited':
            raise HTTPException(429, 'Service busy; retry with backoff', headers={'Retry-After': '1'})
        return asdict(response)

    @api.post('/query/stream')
    async def stream_rag(req: QueryRequest):
        pipeline = await current()
        async def events():
            async for event in pipeline.stream_query(**req.model_dump()):
                yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"
        return StreamingResponse(events(), media_type='text/event-stream',
                                 headers={'X-Streaming-Mode': 'buffered-response-replay'})

    @api.get('/documents')
    async def list_documents():
        return await asyncio.to_thread(kb.list_documents)

    @api.get('/documents/{doc_id}')
    async def get_document(doc_id: str):
        pipeline = await current()
        for doc in pipeline.documents:
            if doc['doc_id'] == doc_id:
                return {'document': doc, 'corpus_revision': pipeline.corpus_revision}
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
                and provenance.get('qa_sha256') == sha256_file(cfg.golden_qa_path))
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

    @api.get('/stats')
    async def get_stats():
        pipeline = await current()
        qa = load_golden_qa(cfg.golden_qa_path)
        return {'telemetry': pipeline.get_stats(),
                'domains': sorted({d['domain'] for d in pipeline.documents}),
                'sample_queries': [{'question_id': q['question_id'], 'category': q['category'],
                                    'question': q['question'],
                                    'paraphrase': next(iter(q.get('paraphrase_variants', [])), q['question'])}
                                   for q in qa[:24]]}

    app.include_router(api)
    return app
