"""Probe a frozen feature snapshot without modifying the live project."""

import asyncio
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import httpx

from rag_platform.awesome_rag import AutoRAGTuner, LightRAGEntityGraph
from rag_platform.config import RAGConfig
from rag_platform.data_gen import load_corpus_documents, load_golden_qa
from rag_platform.evaluation import run_ablation_experiments
from rag_platform.pipeline import EnterpriseRAGPipeline
from rag_platform.server import create_app

OUT = Path(__file__).resolve().parent
SNAP = OUT/'snapshot'
SOURCE = Path(inspect.getfile(EnterpriseRAGPipeline)).parent
RESULT = {'source_root':str(SOURCE),'scope':'frozen snapshot; local generation; no network or paid models',
          'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob('*.py')}}


def cfg(**kwargs):
    return RAGConfig(project_root=SNAP, corpus_path=SNAP/'data/corpus/documents.jsonl',
                     golden_qa_path=SNAP/'data/eval/golden_qa.jsonl', reports_dir=OUT/'fresh_experiments',
                     index_dir=OUT/'indices', persist_index=False, llm_mode='local',
                     embedding_backend='local_svd', **kwargs)


async def main():
    docs = load_corpus_documents(cfg().corpus_path)
    qa = load_golden_qa(cfg().golden_qa_path)
    durations = []
    def timed_graph(chunks):
        start = time.perf_counter()
        graph = LightRAGEntityGraph(chunks)
        durations.append((time.perf_counter()-start)*1000)
        return graph
    with patch('rag_platform.pipeline.LightRAGEntityGraph',side_effect=timed_graph):
        plain = EnterpriseRAGPipeline(cfg(),documents=docs)
    RESULT['graph_when_disabled'] = {
        'enable_graph_rag':plain.config.enable_graph_rag,
        'constructor_calls':len(durations),'construction_ms':durations,
        'entities':len(plain.graph_index.entity_to_chunks),
        'directed_neighbor_entries':sum(len(n) for n in plain.graph_index.neighbors.values()),
        'ordinary_prose_entities':sorted(LightRAGEntityGraph._extract_entities('the and ordinary prose'))}
    RESULT['contextual_header'] = {'indexed_chunks':len(plain.chunks),
                                   'chunks_with_enricher_metadata':sum(bool(c.metadata.get('contextual_header')) for c in plain.chunks)}

    advanced = EnterpriseRAGPipeline(cfg(enable_crag=True,enable_graph_rag=True,
                                          enable_query_expansion=True),documents=docs)
    RESULT['incorrect_router_cases'] = []
    for count in (8,10,12,14,18,24):
        query = 'SOC2-CC6.7 ' + ' '.join(f'zxqnonce{i}' for i in range(count))
        response = await advanced.query(query,use_cache=False)
        item = {'query':query,'crag_status':response.crag_status,'refused':response.refused,
                'confidence':response.confidence,'citations':response.citations}
        RESULT['incorrect_router_cases'].append(item)
        if response.crag_status == 'INCORRECT' and not response.refused:
            break

    question = 'What does error code ERR_QDRANT_5032 indicate and how is it recovered?'
    first = await plain.query(question,use_cache=True)
    plain.config.enable_crag = True
    cached = await plain.query(question,use_cache=True)
    fresh = await plain.query(question,use_cache=False)
    RESULT['feature_toggle_cache'] = {'first_status':first.crag_status,'flag_after_toggle':True,
                                      'cached_status':cached.crag_status,'cache_tier':cached.cache_tier,
                                      'uncached_status':fresh.crag_status,
                                      'note':'Programmatic hot configuration change; UI currently does not expose these flags'}

    import rag_platform.pipeline as module
    real_fuse = module.fuse_trimodal_rrf
    observed = []
    def capture_fuse(*args,**kwargs):
        observed.append({key:kwargs[key] for key in ('dense_weight','sparse_weight','graph_weight','rrf_k')})
        return real_fuse(*args,**kwargs)
    customized = EnterpriseRAGPipeline(cfg(enable_graph_rag=True,dense_weight=.9,sparse_weight=.1),documents=docs)
    with patch('rag_platform.pipeline.fuse_trimodal_rrf',side_effect=capture_fuse):
        await customized.query(question,use_cache=False)
    tuned = AutoRAGTuner.tune_rrf_weights(customized.retriever,customized.graph_index,customized.reranker,qa)
    RESULT['tuner'] = {'configured_dense_weight':.9,'configured_sparse_weight':.1,
                       'effective_trimodal_weights':observed,'graph_weight_configurable':hasattr(customized.config,'graph_weight'),
                       'selection':tuned,
                       'applied_dense_weight_after_tuning':customized.config.dense_weight}
    try:
        AutoRAGTuner.tune_rrf_weights(customized.retriever,customized.graph_index,customized.reranker,[])
    except Exception as exc:
        RESULT['empty_tuner'] = {'exception':type(exc).__name__,'message':str(exc)}

    compressed = EnterpriseRAGPipeline(cfg(enable_contextual_compression=True),documents=docs)
    RESULT['natural_compression'] = []
    for item in qa[:12]:
        response = await compressed.query(item['question'],use_cache=False)
        RESULT['natural_compression'].append({'question':item['question'],
                                              'ratio':response.stage_timings_ms.get('compression_ratio'),
                                              'lexical_grounding_proxy':response.lexical_grounding_proxy})
        if response.lexical_grounding_proxy is not None and response.lexical_grounding_proxy < .99:
            break
    def strong_extract(query,parent_text,max_sentences=4):
        line = parent_text.splitlines()[0]
        return line, len(line)/max(1,len(parent_text))
    with patch.object(compressed.compressor,'compress',side_effect=strong_extract):
        response = await compressed.query(question,use_cache=False)
    RESULT['controlled_compression'] = {'answer':response.answer,
                                         'compressed_evidence':[e.parent_text for e in response.evidence],
                                         'lexical_grounding_proxy':response.lexical_grounding_proxy,
                                         'note':'Compressor forced to retain only existing first line. Local synthesis still reads original candidates.'}

    with tempfile.TemporaryDirectory(dir=OUT,prefix='empty-') as temp:
        folder=Path(temp)
        empty=RAGConfig(project_root=folder,corpus_path=folder/'corpus.jsonl',golden_qa_path=folder/'qa.jsonl',
                        reports_dir=folder/'reports',index_dir=folder/'index',bootstrap_synthetic=False,
                        persist_index=False,llm_mode='local',embedding_backend='local_svd')
        app=create_app(empty)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app,raise_app_exceptions=False),base_url='http://test') as client:
            response=await client.post('/api/v1/autorag/tune')
        RESULT['empty_tuner_http']={'status':response.status_code,'body':response.text}

    ablation = await run_ablation_experiments(cfg(),save_reports=True)
    RESULT['seven_architectures'] = [{'name':e['spec']['name'],
                                     'recall_at_5':e['metrics']['recall_at_5']['mean'],
                                     'mrr':e['metrics']['mrr']['mean'],
                                     'ndcg_at_5':e['metrics']['ndcg_at_5']['mean'],
                                     'keyword_coverage':e['metrics']['keyword_coverage']['mean']}
                                    for e in ablation['experiments']]
    RESULT['advanced_real_domain'] = []
    from rag_platform.evaluation import evaluate_pipeline
    domain_docs=load_corpus_documents(SNAP/'data/python_support/documents.jsonl')
    domain_qa=load_golden_qa(SNAP/'data/python_support/qa_test.jsonl')
    for enabled in (False,True):
        domain=EnterpriseRAGPipeline(cfg(enable_crag=enabled,enable_graph_rag=enabled,
                                        enable_query_expansion=enabled),documents=domain_docs)
        evaluation=await evaluate_pipeline(domain,domain_qa)
        RESULT['advanced_real_domain'].append({'enabled':enabled,'metrics':evaluation['metrics']})
    (OUT/'probe_results.json').write_text(json.dumps(RESULT,indent=2,ensure_ascii=False)+'\n')
    compact={k:v for k,v in RESULT.items() if k not in ('source_sha256','tuner','controlled_compression','advanced_real_domain')}
    print(json.dumps(compact,indent=2,ensure_ascii=False))
    print('Best tuner config:',tuned['best_config'])
    print('Local compression proxy:',RESULT['controlled_compression']['lexical_grounding_proxy'])
    print('Domain comparisons:',[{**{'enabled':r['enabled']},**{k:r['metrics'][k]['mean'] for k in ('recall_at_5','mrr','ndcg_at_5','keyword_coverage')}} for r in RESULT['advanced_real_domain']])


if __name__=='__main__':
    asyncio.run(main())
