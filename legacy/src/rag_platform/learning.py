"""A small curriculum and query-level, single-variable experiments."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
import uuid

from .bootstrap import atomic_text
from .compute import run_cpu
from .pipeline import EnterpriseRAGPipeline
from .security import redact_pii

QUESTION = 'What is the current EU restore recovery target for Cedar?'
LESSONS = [
    dict(id='01', title='跑通一条 RAG 链路', concept='问题先找到资料，再根据证据回答。',
         task='关闭缓存与重排，运行示例问题，打开完整证据并找到支持答案的原句。',
         question=QUESTION, variable='retrieval_mode', a='bm25', b='bm25', action='playground',
         check='能解释原文、检索块、回答与引用分别是什么；指出默认回答是本地抽取式。', source='pipeline.py'),
    dict(id='02', title='分块：匹配片段与回答上下文', concept='分块影响匹配粒度，也影响限制条件是否完整。这里的长度单位是字符。',
         task='先预览同一原文的 fixed / parent-child 分块，再用同一问题对照。',
         question=QUESTION, variable='chunking_strategy', a='fixed', b='parent_child', action='lab',
         check='指出一个边界变化，并解释 child 与 parent 的用途；不假定某种策略总更好。', source='chunking.py'),
    dict(id='03', title='BM25 与向量检索', concept='BM25 匹配词项；默认向量来自语料内 TF-IDF + SVD，还不是预训练语义模型。',
         task='固定分块，比较精确术语和你自己写的改述；观察第一条证据为何变化。',
         question='When aggregating successful coroutines, are results listed by arrival time or by the original input order?',
         variable='retrieval_mode', a='bm25', b='dense', action='lab',
         check='能说出两种检索各自的失败例子，区分算法输出分数和事实正确率。', source='retrieval.py'),
    dict(id='04', title='RRF：融合排名', concept='RRF 融合名次；BM25 与余弦分数不能直接相加比较。',
         task='只把 BM25 换成 hybrid_rrf，记录新增或消失的文档及顺序。',
         question=QUESTION, variable='retrieval_mode', a='bm25', b='hybrid_rrf', action='lab',
         check='手算两个候选的加权 RRF，说明融合有没有帮助这条问题。', source='retrieval.py'),
    dict(id='05', title='重排：从候选到最终证据', concept='检索负责找候选，重排负责重新排序。默认是人工特征重排。',
         task='固定 hybrid_rrf，只切换重排开关；记录排名、引用和回答变化。',
         question=QUESTION, variable='use_reranker', a=False, b=True, action='lab',
         check='能区分特征重排与真正的 cross-encoder，不把多一层处理当作效果保证。', source='reranker.py'),
    dict(id='06', title='拒答：覆盖率与错误的取舍', concept='启发式相关性分数不是正确概率；阈值升高可能增加误拒答。',
         task='对有答案、无答案和短问题分别比较 0.22 / 0.50 阈值。',
         question='What is the current Cedar US renewal fee in dollars?',
         variable='min_confidence_threshold', a=.22, b=.50, action='lab',
         check='人工判断该不该拒答，记录错误作答和错误拒答；再决定是否学习校准。', source='study.py'),
    dict(id='07', title='评价：先检查失败，再汇总指标', concept='命中文档不等于回答正确；多跳题需要所有必要证据。',
         task='只改变 top-k，核对两个必要事实；另写至少五道问题并人工标注支持句。',
         question='Under the 2026 Cedar EU policies, give both the restore recovery target and export retention period.',
         variable='top_k', a=1, b=5, action='lab',
         check='交出一次单变量记录和一个失败案例；能够解释为何不能在测试集上挑参数。', source='evaluation.py'),
]


def curriculum():
    return {'goal': '理解链路 → 单变量实验 → 证据核查 → 失败分析', 'lessons': LESSONS,
            'default_profile': 'offline_learning',
            'advanced': ['预训练 encoder 与 cross-encoder', '压缩、规则扩展与共现图',
                         '开发集校准', '缓存与并发', 'Qdrant 与交付演示']}


def validate_pair(variable, a, b):
    choices = {'chunking_strategy': {'fixed','recursive','semantic','parent_child'},
               'retrieval_mode': {'bm25','dense','hybrid_rrf'}}
    if variable in choices:
        valid = all(isinstance(value, str) and value in choices[variable] for value in (a,b))
    elif variable == 'use_reranker':
        valid = all(type(value) is bool for value in (a,b))
    elif variable == 'top_k':
        valid = all(type(value) is int and 1 <= value <= 10 for value in (a,b))
    elif variable == 'min_confidence_threshold':
        valid = all(type(value) in (int,float) and math.isfinite(value) and 0 <= value <= 1 for value in (a,b))
    else:
        valid = False
    if not valid or a == b:
        raise ValueError('Choose two different, valid values for one learning variable')


async def compare(snapshot, query, variable, a, b, hypothesis='', save_report=False):
    validate_pair(variable, a, b)
    # A lesson is independent of the service's optional deployment/model profile.
    base = replace(snapshot.config, chunking_strategy='parent_child', retrieval_mode='bm25',
        use_reranker=False, use_cache=False, enable_singleflight=False, persist_index=False,
        embedding_backend='local_svd', embedding_revision='', vector_backend='memory',
        reranker_backend='features', llm_mode='local', use_calibration=False,
        enable_crag=False, enable_graph_rag=False, enable_query_expansion=False,
        enable_contextual_compression=False, enable_contextual_headers=False,
        expansion_rules_path=None, top_k_final=3, min_confidence_threshold=.22)
    field = 'top_k_final' if variable == 'top_k' else variable
    results = []
    for label, value in [('A',a),('B',b)]:
        changes = {field:value}
        if variable == 'use_reranker':
            changes['retrieval_mode'] = 'hybrid_rrf'
        config = replace(base, **changes)
        pipeline = await run_cpu(EnterpriseRAGPipeline, config, snapshot.documents, workers=base.cpu_workers)
        try:
            response = await pipeline.query(query, use_cache=False)
            results.append({'label':label, 'value':value, 'chunks':len(pipeline.chunks),
                            'embedding_dimension_actual':pipeline.retriever.dense_index.embeddings.shape[1],
                            'response':asdict(response)})
        finally:
            await pipeline.aclose()
    settings = {'chunking_strategy':base.chunking_strategy, 'retrieval_mode':base.retrieval_mode,
                'use_reranker':base.use_reranker, 'top_k':base.top_k_final,
                'min_confidence_threshold':base.min_confidence_threshold,
                'embedding':'TF-IDF + SVD', 'generation':'local extractive',
                'cache':False, 'calibration':False,
                'chunk_sizes_characters':{'fixed':base.fixed_chunk_size,'overlap':base.fixed_overlap,
                                         'parent':base.parent_max_chars,'child':base.child_max_chars},
                'embedding_dimension_requested':base.embedding_dim,
                'rrf':{'dense_weight':base.dense_weight,'sparse_weight':base.sparse_weight,'k':base.rrf_k}}
    settings.pop(variable, None)
    if variable == 'use_reranker':
        settings['retrieval_mode'] = 'hybrid_rrf'
    result = {'experiment_id':uuid.uuid4().hex, 'time':datetime.now(timezone.utc).isoformat(),
              'variable':variable, 'hypothesis':redact_pii(hypothesis)[0],
              'corpus_revision':snapshot.corpus_revision, 'controlled_settings':settings, 'results':results,
              'notes':['Single-query exploration; no general quality or throughput conclusion.',
                       'Relevance-score scales differ across retrieval methods; inspect the evidence first.',
                       'Chunking changes the fitted SVD representation as a consequence of that choice.',
                       'Check the supporting sentences and record your own observation.'], 'saved':False}
    if save_report:
        path = base.reports_dir/'learning_lab'/(result['experiment_id']+'.json')
        result['saved'] = True
        result['report_path'] = str(path)
        await run_cpu(atomic_text, path, json.dumps(result,ensure_ascii=False,indent=2)+'\n',workers=base.cpu_workers)
    return result


def chunk_preview(snapshot, doc_id, strategy):
    from .chunking import build_chunker
    if strategy not in {'fixed','recursive','parent_child'}:
        raise ValueError('Preview supports fixed, recursive or parent_child; semantic chunking is available in the comparison lab')
    document = snapshot.document_by_id.get(doc_id)
    if document is None:
        raise KeyError(doc_id)
    config = snapshot.config
    chunker = build_chunker(strategy, fixed_chunk_size=config.fixed_chunk_size,
        fixed_overlap=config.fixed_overlap, recursive_max_chars=config.recursive_max_chars,
        parent_max_chars=config.parent_max_chars, child_max_chars=config.child_max_chars)
    chunks = chunker.chunk_documents([document])
    return {'doc_id':doc_id, 'strategy':strategy, 'unit':'characters',
            'original_text':document['text'], 'total_chunks':len(chunks),
            'chunks':[asdict(chunk) for chunk in chunks[:50]], 'truncated':len(chunks)>50}
