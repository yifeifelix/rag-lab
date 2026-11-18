
import asyncio,json,os
from pathlib import Path
from dataclasses import replace
from rag_platform.config import RAGConfig
from rag_platform.pipeline import EnterpriseRAGPipeline
from rag_platform.data_gen import load_corpus_documents
out=Path(os.environ['FEATURE_REVIEW_OUTPUT']);snap=out/'snapshot'
cfg=RAGConfig(project_root=snap,corpus_path=snap/'data/corpus/documents.jsonl',golden_qa_path=snap/'data/eval/golden_qa.jsonl',persist_index=False,llm_mode='local',embedding_backend='local_svd')
docs=load_corpus_documents(cfg.corpus_path)
async def main():
 p=EnterpriseRAGPipeline(cfg,documents=docs)
 q='What does error code ERR_QDRANT_5032 indicate and how is it recovered?'
 normal=await p.query(q,use_cache=False)
 compressed=await EnterpriseRAGPipeline(replace(cfg,enable_contextual_compression=True),documents=docs).query(q,use_cache=False)
 cq='Why does compressing vector embeddings to 8-bit integers save memory without hurting search quality?'
 initial=p.crag.evaluate(cq,p.retriever.retrieve(cq,top_k_candidates=36),p.retriever.bm25_index.vocab)
 advanced=await EnterpriseRAGPipeline(replace(cfg,enable_crag=True,enable_query_expansion=True,enable_graph_rag=True),documents=docs).query(cq,use_cache=False)
 result={'natural_compression':{'answers_identical':normal.answer==compressed.answer,'baseline_proxy':normal.lexical_grounding_proxy,'compressed_proxy':compressed.lexical_grounding_proxy,'compression_ratio':compressed.stage_timings_ms.get('compression_ratio'),'answer':compressed.answer,'compressed_evidence':[e.parent_text for e in compressed.evidence]},'crag_correct_path':{'initial_assessment':initial.status,'initial_confidence':initial.top_confidence,'reported_final_status':advanced.crag_status,'expanded_terms':advanced.query_expanded_terms}}
 (out/'extra_results.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:{a:b for a,b in v.items() if a not in ('answer','compressed_evidence')} for k,v in result.items()},indent=2))
asyncio.run(main())
