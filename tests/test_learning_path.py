from dataclasses import asdict
from html.parser import HTMLParser
import json
from pathlib import Path

import httpx
import pytest

from rag_platform.config import RAGConfig
from rag_platform.learning import compare, validate_pair
from rag_platform.pipeline import EnterpriseRAGPipeline
from rag_platform.server import create_app


@pytest.mark.asyncio
async def test_learning_comparison_changes_only_the_selected_factor(monkeypatch,tmp_path):
    source = EnterpriseRAGPipeline(RAGConfig())
    source.config.calibration_path=tmp_path/'not-a-calibration.json'
    source.config.calibration_path.write_text('{"unrelated":"must not be read"}')
    source.config.llm_mode='openai'
    source.config.embedding_backend='sentence_transformers'
    import rag_platform.learning as learning
    real = learning.EnterpriseRAGPipeline
    captured = []
    def construct(config, documents):
        captured.append(asdict(config))
        return real(config, documents)
    monkeypatch.setattr(learning,'EnterpriseRAGPipeline',construct)
    def forbidden(*args,**kwargs):
        raise AssertionError('Learning experiments must not load calibration')
    monkeypatch.setattr('rag_platform.study.ConfidenceCalibration.load',forbidden)
    response = await compare(source,'What does ERR_QDRANT_5032 indicate?','use_reranker',False,True)
    assert {key for key in captured[0] if captured[0][key]!=captured[1][key]} == {'use_reranker'}
    assert all(config['retrieval_mode']=='hybrid_rrf' for config in captured)
    assert all(not config['use_cache'] and not config['use_calibration'] and not config['persist_index'] for config in captured)
    assert all(config['llm_mode']=='local' and config['vector_backend']=='memory' for config in captured)
    assert response['controlled_settings']['calibration'] is False
    assert response['saved'] is False
    assert all(row['response']['cache_tier']=='miss' for row in response['results'])
    assert source.config.use_cache is True
    await source.aclose()


def test_comparison_rejects_multiple_or_invalid_controls():
    for args in [('top_k',True,5),('top_k',1,1000),('min_confidence_threshold',.2,float('nan')),
                 ('llm_mode','local','openai'),('retrieval_mode','bm25','bm25')]:
        with pytest.raises(ValueError): validate_pair(*args)


@pytest.mark.asyncio
async def test_course_preview_and_readonly_lab_preserve_source():
    cfg=RAGConfig(admin_token='owner-example',tester_token='learner-example')
    app=create_app(cfg)
    kb=app.state.knowledge_base
    before=cfg.corpus_path.read_bytes()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
        headers={'Authorization':'Bearer learner-example'}
        course=(await client.get('/api/v1/learning',headers=headers)).json()
        assert len(course['lessons'])==7 and course['lessons'][0]['action']=='playground'
        doc_id=kb.current().documents[0]['doc_id']
        preview=await client.get('/api/v1/learning/chunks',params={'doc_id':doc_id,'strategy':'parent_child'},headers=headers)
        assert preview.status_code==200
        assert preview.json()['original_text']==kb.current().document_by_id[doc_id]['text']
        assert preview.json()['unit']=='characters'
        payload={'query':'What does ERR_QDRANT_5032 indicate?','variable':'retrieval_mode','a':'bm25','b':'dense'}
        result=await client.post('/api/v1/learning/compare',headers=headers,json=payload)
        assert result.status_code==200,result.text
        assert not result.json()['saved'] and len(result.json()['results'])==2
        assert cfg.corpus_path.read_bytes()==before
        assert not (cfg.reports_dir/'learning_lab').exists()
        assert (await client.post('/api/v1/learning/compare',headers=headers,json={**payload,'save_report':True})).status_code==403
        saved=await client.post('/api/v1/learning/compare',headers={'Authorization':'Bearer owner-example'},
             json={**payload,'save_report':True,'hypothesis':'Contact alice@example.com'})
        assert saved.status_code==200,saved.text
        path=Path(saved.json()['report_path'])
        assert path.is_relative_to(cfg.reports_dir/'learning_lab')
        assert 'alice@example.com' not in path.read_text()
        assert json.loads(path.read_text())['saved'] is True
    await kb.aclose()


class Page(HTMLParser):
    def __init__(self):
        super().__init__();self.attrs=[];self.select=None;self.defaults={}
    def handle_starttag(self,tag,attrs):
        value=dict(attrs);self.attrs.append((tag,value))
        if tag=='select':self.select=value.get('id')
        if tag=='option' and 'selected' in value:self.defaults[self.select]=value['value']
    def handle_endtag(self,tag):
        if tag=='select':self.select=None


@pytest.mark.asyncio
async def test_home_starts_with_learning_and_lightweight_query_defaults():
    app=create_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
        page=Page();page.feed((await client.get('/')).text)
    active=[value['id'] for tag,value in page.attrs if tag=='section' and value.get('class')=='panel active']
    assert active==['tab-learning']
    assert page.defaults['q-mode']=='bm25' and page.defaults['q-cache']=='false' and page.defaults['q-rerank']=='false'
    assert all('hidden' in value for tag,value in page.attrs if 'data-advanced' in value)
    await app.state.knowledge_base.aclose()
