import asyncio
from dataclasses import asdict
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from rag_platform.cache import L2SemanticCache
from rag_platform.config import RAGConfig
from rag_platform.knowledge_base import KnowledgeBase
from rag_platform.pipeline import EnterpriseRAGPipeline, compute_lexical_grounding
from rag_platform.server import create_app


def isolated_config(tmp_path, **kwargs):
    return RAGConfig(project_root=tmp_path, corpus_path=tmp_path/'corpus.jsonl',
                     golden_qa_path=tmp_path/'qa.jsonl', reports_dir=tmp_path/'reports',
                     index_dir=tmp_path/'index', bootstrap_synthetic=False, **kwargs)


def test_semantic_eviction_never_changes_other_scope_payload():
    cache = L2SemanticCache(max_size=2)
    a, b = np.eye(2, dtype=np.float32)
    cache.set(a, 'A', 'A_old')
    cache.set(b, 'B', 'B_expected')
    cache.set(a, 'A', 'A_new')
    assert cache.get(b, 'B')[0] == 'B_expected'
    assert cache.get(a, 'A')[0] == 'A_new'


def test_expired_best_match_does_not_hide_live_match(monkeypatch):
    import rag_platform.cache as module
    clock = [100.0]
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    cache = L2SemanticCache(ttl_seconds=10)
    vector = np.array([1., 0.], dtype=np.float32)
    cache.set(vector, 'A', 'expired')
    clock[0] = 105
    cache.set(vector, 'A', 'live')
    clock[0] = 111
    assert cache.get(vector, 'A')[0] == 'live'


@pytest.mark.asyncio
async def test_pii_absent_from_responses_caches_and_singleflight():
    pipeline = EnterpriseRAGPipeline(RAGConfig(persist_index=False, simulated_llm_latency_ms=5))
    query = 'My email alice@example.com, what does ERR_QDRANT_5032 mean?'
    responses = await asyncio.gather(*(pipeline.query(query) for _ in range(3)))
    responses.append(await pipeline.query(query))
    payloads = [p for _, p in pipeline.l1_cache._store.values()]
    payloads += [e.payload for e in pipeline.l2_cache._entries]
    assert 'alice@example.com' not in json.dumps([asdict(r) for r in responses + payloads])
    assert all(r.pii_redacted for r in responses)
    assert not pipeline.singleflight._inflight


def test_lexical_proxy_explicitly_does_not_measure_correctness():
    evidence = [SimpleNamespace(parent_text='Alice paid Bob 100 dollars.', title='Payments')]
    assert compute_lexical_grounding('Bob paid Alice 100 dollars.', evidence, False) == 1.0
    assert compute_lexical_grounding('Cannot answer', [], True) is None


@pytest.mark.asyncio
async def test_number_changes_cannot_reuse_semantic_cache():
    p = EnterpriseRAGPipeline(RAGConfig(persist_index=False))
    await p.query('What is the uptime target in SLA-TIER0-99.99?')
    result = await p.query('What is the uptime target in SLA-TIER0-99.95?')
    assert result.cache_tier == 'miss'


@pytest.mark.asyncio
async def test_document_lifecycle_refresh_persistence_and_delete(tmp_path):
    cfg = isolated_config(tmp_path)
    app = create_app(cfg)
    kb = app.state.knowledge_base
    other_worker = KnowledgeBase(cfg)
    document = {'doc_id':'ORBIT-SVC-101', 'title':'Orbit support escalation', 'domain':'support',
                'source':'https://example.org/orbit', 'version':'1',
                'text':'The ORBIT-SVC-101 support escalation recovery window is 17 minutes. '
                       'An operator restores service from a verified support snapshot.'}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        created = await client.post('/api/v1/documents', json=document)
        assert created.status_code == 200
        revision = created.json()['corpus_revision']
        query = {'query':'What is the ORBIT-SVC-101 support escalation recovery window?'}
        first = (await client.post('/api/v1/query', json=query)).json()
        assert '17' in first['answer']
        assert first['evidence'][0]['source'] == document['source']
        assert (await client.post('/api/v1/query', json=query)).json()['cache_tier'] == 'l1_exact'
        assert len(other_worker.current().documents) == 1
        restarted = KnowledgeBase(cfg)
        assert restarted.current().index_loaded_from_disk
        assert (await restarted.current().query(**query)).answer == first['answer']
        document['text'] = document['text'].replace('17 minutes', '43 minutes')
        document['version'] = '2'
        updated = await client.put('/api/v1/documents/ORBIT-SVC-101', json=document,
                                   headers={'If-Match':revision})
        assert updated.status_code == 200
        stale = await client.put('/api/v1/documents/ORBIT-SVC-101', json=document,
                                 headers={'If-Match':revision})
        assert stale.status_code == 409
        second = (await client.post('/api/v1/query', json=query)).json()
        assert second['cache_tier'] == 'miss'
        assert '43' in second['answer'] and '17' not in second['answer']
        assert '43 minutes' in other_worker.current().documents[0]['text']
        assert len(list(cfg.index_dir.glob('*.npz'))) == 1
        assert (await client.delete('/api/v1/documents/ORBIT-SVC-101')).status_code == 200
        assert (await client.post('/api/v1/query', json=query)).json()['refused']
        assert not other_worker.current().documents
        assert not KnowledgeBase(cfg).current().documents
        assert (await client.get('/api/v1/documents/ORBIT-SVC-101')).status_code == 404
        events = [json.loads(line) for line in kb.events_path.read_text().splitlines()]
        assert [e['operation'] for e in events] == ['create','update','delete']
        assert all('text' not in e for e in events)


def test_failed_build_preserves_corpus_and_current_pipeline(tmp_path, monkeypatch):
    cfg = isolated_config(tmp_path)
    kb = KnowledgeBase(cfg)
    before = cfg.corpus_path.read_bytes()
    old = kb.current()
    def fail(*args, **kwargs):
        raise RuntimeError('model not available')
    monkeypatch.setattr('rag_platform.knowledge_base.EnterpriseRAGPipeline', fail)
    with pytest.raises(RuntimeError):
        kb.mutate({'doc_id':'DOC-1','title':'Title','domain':'support','text':'some new document contents'})
    assert cfg.corpus_path.read_bytes() == before
    assert kb.current() is old


@pytest.mark.asyncio
async def test_owner_token_and_real_http_rate_limit_status(tmp_path, monkeypatch):
    app = create_app(isolated_config(tmp_path, admin_token='test-owner-token'))
    monkeypatch.setattr(app.state.knowledge_base.current().rate_limiter, 'allow', lambda: False)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
        assert (await client.get('/api/v1/documents')).status_code == 401
        assert (await client.post('/api/v1/documents', json={})).status_code == 401
        headers = {'Authorization':'Bearer test-owner-token'}
        assert (await client.get('/api/v1/documents',headers=headers)).status_code == 200
        response = await client.post('/api/v1/query', json={'query':'service recovery'},headers=headers)
        assert response.status_code == 429
        assert response.headers['retry-after'] == '1'


@pytest.mark.asyncio
async def test_remote_failure_is_visible_and_not_cached(tmp_path, monkeypatch):
    p = EnterpriseRAGPipeline(RAGConfig(persist_index=False, llm_mode='openai'))
    async def fail(*args):
        raise RuntimeError('provider unavailable')
    monkeypatch.setattr(p, '_call_remote_llm', fail)
    r = await p.query('What does ERR_QDRANT_5032 indicate?')
    assert r.degraded_mode and r.generation_mode == 'extractive_fallback'
    assert not len(p.l1_cache) and not len(p.l2_cache)


@pytest.mark.asyncio
async def test_cloud_request_uses_sanitized_prompt_and_reports_usage(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-not-a-real-key')
    monkeypatch.setenv('RAG_INPUT_USD_PER_MILLION', '1')
    monkeypatch.setenv('RAG_OUTPUT_USD_PER_MILLION', '2')
    real_client = httpx.AsyncClient
    def respond(request):
        assert 'alice@example.com' not in request.content.decode()
        payload = json.loads(request.content)
        context = json.loads(payload['messages'][1]['content'])
        assert any(row['doc_id'] == 'CLOUD-ARCH-002' for row in context['evidence'])
        return httpx.Response(200, json={
            'choices':[{'message':{'content':'Shard quorum loss [CLOUD-ARCH-002]'}}],
            'usage':{'prompt_tokens':100,'completion_tokens':25}})
    monkeypatch.setattr('rag_platform.pipeline.httpx.AsyncClient',
                        lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs))
    pipeline = EnterpriseRAGPipeline(RAGConfig(persist_index=False, llm_mode='openai'))
    result = await pipeline.query('My email alice@example.com, what does ERR_QDRANT_5032 mean?')
    assert result.generation_mode == 'openai' and not result.degraded_mode
    assert result.token_usage == {'input_tokens':100, 'output_tokens':25}
    assert result.estimated_cost_usd == pytest.approx(0.00015)
    assert result.citations == ['CLOUD-ARCH-002']
    cached = await pipeline.query('My email alice@example.com, what does ERR_QDRANT_5032 mean?')
    assert cached.token_usage == {} and cached.estimated_cost_usd == 0


def test_pretrained_missing_dependency_fails_instead_of_silent_fallback(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'sentence_transformers', None)
    with pytest.raises(RuntimeError, match='pretrained'):
        EnterpriseRAGPipeline(RAGConfig(persist_index=False, embedding_backend='sentence_transformers'))


@pytest.mark.asyncio
async def test_remote_clients_require_owner_token(tmp_path):
    app = create_app(isolated_config(tmp_path))
    transport = httpx.ASGITransport(app=app, client=('203.0.113.10', 12345))
    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
        assert (await client.get('/api/v1/documents')).status_code == 403


def test_dotenv_precedence_and_data_volume_cache_routing(tmp_path, monkeypatch):
    from rag_platform.runtime import configure_runtime
    import os
    import tempfile
    monkeypatch.setattr(tempfile, 'tempdir', tempfile.tempdir)
    (tmp_path/'.env').write_text('RAG_REVIEW_SAMPLE="from file"\n')
    monkeypatch.delenv('RAG_REVIEW_SAMPLE', raising=False)
    monkeypatch.setenv('TMPDIR', '/private/tmp')
    paths = configure_runtime(tmp_path)
    assert os.environ['RAG_REVIEW_SAMPLE'] == 'from file'
    assert paths['TMPDIR'] == str(tmp_path/'.cache/tmp')
    monkeypatch.setenv('RAG_REVIEW_SAMPLE', 'exported value')
    configure_runtime(tmp_path)
    assert os.environ['RAG_REVIEW_SAMPLE'] == 'exported value'


def test_unmeasured_metrics_are_null_not_perfect_or_zero():
    from rag_platform.evaluation import bootstrap_ci
    assert bootstrap_ci([]) == {'mean':None, 'ci_low':None, 'ci_high':None, 'n':0}
