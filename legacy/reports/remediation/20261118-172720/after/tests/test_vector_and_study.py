from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from rag_platform.chunking import Chunk
from rag_platform.config import RAGConfig
from rag_platform.knowledge_base import KnowledgeBase
from rag_platform.pipeline import EnterpriseRAGPipeline
from rag_platform.reranker import CrossEncoderReranker
from rag_platform.retrieval import RetrievedCandidate
from rag_platform.study import ConfidenceCalibration, CalibrationMismatch, RRFGridSearch, load_dev, calibrate
from rag_platform.vector_store import IncrementalBM25Index


class FixedEncoder:
    revision = 'a' * 40
    embedding_dim = 8
    svd = None
    def __init__(self):
        self.encoded = []
    def fit(self, texts):
        return self
    def encode(self, texts):
        self.encoded.extend(texts)
        values = np.array([[value / 255 for value in hashlib.sha256(text.encode()).digest()[:8]] for text in texts], dtype=np.float32)
        return values / np.linalg.norm(values, axis=1, keepdims=True)
    def encode_single(self, text):
        return self.encode([text])[0]


class QdrantContract:
    def __init__(self):
        self.points = {}
        self.created = False
        self.upserts = []
        self.deleted = []
        self.query_filters = []
    def handle(self, request):
        path = request.url.path
        body = json.loads(request.content) if request.content else None
        if request.method == 'GET':
            if not self.created:
                return httpx.Response(404, json={})
            result = {'config':{'params':{'vectors':{'size':8,'distance':'Cosine'}}}}
        elif path.endswith('/points/scroll'):
            result = {'points':[{'id':key} for key in self.points], 'next_page_offset':None}
        elif path.endswith('/points/query'):
            self.query_filters.append(body['filter'])
            allowed = set(body['filter']['must'][0]['has_id'])
            domain = next((entry['match']['value'] for entry in body['filter']['must'] if entry.get('key')=='domain'), None)
            scores = [{'id':key,'score':float(np.dot(value['vector'],body['query']))}
                      for key,value in self.points.items() if key in allowed and (not domain or value['payload']['domain']==domain)]
            result = {'points':sorted(scores,key=lambda value:-value['score'])[:body['limit']]}
        elif path.endswith('/points/delete'):
            for key in body['points']:
                self.points.pop(key, None)
                self.deleted.append(key)
            result = {'status':'completed'}
        elif path.endswith('/points'):
            self.upserts.append(len(body['points']))
            for point in body['points']:
                self.points[point['id']] = point
            result = {'status':'completed'}
        else:
            self.created = True
            result = True
        return httpx.Response(200,json={'result':result})


def test_qdrant_incremental_update_delete_restart_and_old_snapshot(tmp_path, monkeypatch):
    encoder = FixedEncoder()
    monkeypatch.setattr('rag_platform.pipeline.build_embedder',lambda config:encoder)
    contract = QdrantContract()
    real = httpx.Client
    monkeypatch.setattr('rag_platform.vector_store.httpx.Client', lambda **kwargs: real(transport=httpx.MockTransport(contract.handle), **kwargs))
    cfg = RAGConfig(project_root=tmp_path, corpus_path=tmp_path/'corpus.jsonl', golden_qa_path=tmp_path/'qa.jsonl',
                    index_dir=tmp_path/'index', reports_dir=tmp_path/'reports', bootstrap_synthetic=False,
                    embedding_backend='sentence_transformers', embedding_revision='a'*40, vector_backend='qdrant')
    kb = KnowledgeBase(cfg)
    first = {'doc_id':'orbit.eu','title':'Orbit recovery','domain':'EU','version':'1',
             'text':'Orbit EU recovery uses a verified snapshot within seventeen minutes.'}
    second = {'doc_id':'orbit.us','title':'Orbit recovery','domain':'US','version':'1',
              'text':'Orbit US recovery uses a verified snapshot within thirty minutes.'}
    kb.mutate(first)
    kb.mutate(second)
    old = kb.current()
    old_ids = set(old.retriever.dense_index.ids)
    before = len(encoder.encoded)
    first['text'] = first['text'].replace('seventeen','forty three')
    first['version'] = '2'
    kb.mutate(first)
    assert len(encoder.encoded) - before == 1, 'Only the changed chunk should be encoded'
    assert kb.current().retriever.bm25_index.tokenized_chunks == 1
    assert kb.current().update_stats['reused_chunks'] == 1
    active = kb.current().retriever.dense_index
    assert len(contract.points) == 2
    assert old_ids - set(active.ids) <= set(contract.deleted)
    # Old queries still have their immutable local vectors after remote deletion.
    assert old.retriever.dense_index.search_vector(np.ones(8,dtype=np.float32),2)
    found = active.search_vector(np.ones(8,dtype=np.float32),5,'EU')
    assert found and all(active.chunks[index].domain=='EU' for index, _ in found)
    assert set(contract.query_filters[-1]['must'][0]['has_id']) == set(active.ids)
    prior_upserts = len(contract.upserts)
    restart = KnowledgeBase(cfg)
    assert len(contract.upserts) == prior_upserts, 'Restart must reuse existing immutable points'
    kb.mutate(delete_id='orbit.eu')
    assert len(contract.points) == 1
    assert all(point['payload']['doc_id'] != 'orbit.eu' for point in contract.points.values())
    assert all('text' not in point['payload'] for point in contract.points.values())
    active.close()
    restart.current().retriever.dense_index.close()
    kb.current().retriever.dense_index.close()


def test_incremental_bm25_preserves_old_snapshot():
    a = Chunk('a','a','Alpha','support','alpha recovery policy','alpha recovery policy')
    b = Chunk('b','b','Beta','support','beta support policy','beta support policy')
    first = IncrementalBM25Index([a,b])
    updated = replace(a,text='alpha recovery revised')
    second = IncrementalBM25Index([updated,b],previous=first)
    assert second.tokenized_chunks == 1
    assert first.search('revised') == []
    assert second.search('revised')[0][0] == 0
    third = IncrementalBM25Index([b], previous=second)
    assert third.tokenized_chunks == 0 and third.search('alpha') == []
    assert first.search('alpha')


def test_real_cross_encoder_adapter_uses_joint_pairs_and_score_for_gate(monkeypatch):
    calls = []
    class Model:
        def __init__(self, model_name, activation_fn=None, **kwargs):
            assert kwargs['local_files_only'] and not kwargs['trust_remote_code']
            self.model = SimpleNamespace(config=SimpleNamespace(_commit_hash='b'*40))
        def predict(self, pairs, **kwargs):
            calls.extend(pairs)
            return np.array([-8, 8],dtype=np.float32)
    monkeypatch.setitem(sys.modules,'sentence_transformers',SimpleNamespace(CrossEncoder=Model))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(nn=SimpleNamespace(Identity=lambda:None)))
    reranker = CrossEncoderReranker(RAGConfig(reranker_backend='cross_encoder'))
    chunks = [Chunk('a','a','Recovery','support','recovery policy A','recovery policy A'),
              Chunk('b','b','Recovery','support','recovery policy B','recovery policy B')]
    candidates = [RetrievedCandidate(chunk=chunk,dense_score=.5,bm25_score=4,rrf_score=.1,confidence_score=.5) for chunk in chunks]
    result = reranker.rerank('recovery policy',candidates,top_k=2)
    assert len(calls)==2 and all(pair[0]=='recovery policy' for pair in calls)
    assert result[0].rerank_score > result[1].rerank_score
    assert result[0].confidence_score > .99 and result[1].confidence_score < .01


def test_dev_loader_does_not_read_test_questions(tmp_path, monkeypatch):
    cfg = RAGConfig(dev_qa_path=tmp_path/'dev.jsonl',test_qa_path=tmp_path/'test.jsonl')
    cfg.dev_qa_path.write_text(json.dumps({'question_id':'dev1','group_id':'g1','split':'dev','question':'Known dev'})+'\n')
    cfg.test_qa_path.write_text('DO NOT READ THIS DURING TUNING')
    original = Path.read_text
    def read(path, *args, **kwargs):
        assert path != cfg.test_qa_path
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    assert len(load_dev(cfg)) == 1


@pytest.mark.asyncio
async def test_group_calibration_artifact_rejects_changed_config_and_corpus(tmp_path, monkeypatch):
    cfg = RAGConfig(dev_qa_path=tmp_path/'dev.jsonl')
    rows = [{'question_id':f'{group}-{label}','group_id':str(group),'split':'dev','label':label}
            for group in range(6) for label in [0,1]]
    cfg.dev_qa_path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    pipeline = SimpleNamespace(config=cfg,corpus_revision='revision-one')
    def observe(pipeline,item):
        return {'score':.9 if item['label'] else .1, 'supported':item['label'],
                'group_id':item['group_id'],'question_id':item['question_id']}
    monkeypatch.setattr('rag_platform.study.retrieval_observation',observe)
    result = await calibrate(pipeline,rows)
    assert result['oof_brier_score'] == 0
    path = tmp_path/'calibration.json'
    path.write_text(json.dumps(result))
    model = ConfidenceCalibration.load(path,cfg,'revision-one')
    assert model.predict(.9) > model.predict(.1)
    with pytest.raises(CalibrationMismatch):
        ConfidenceCalibration.load(path,replace(cfg,graph_weight=.8),'revision-one')
    with pytest.raises(CalibrationMismatch):
        ConfidenceCalibration.load(path,cfg,'revision-two')
    with pytest.raises(ValueError):
        await calibrate(pipeline,[{**row,'split':'test'} for row in rows])
