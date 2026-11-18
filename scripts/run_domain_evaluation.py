"""Controlled retrieval comparisons and auditable pending human-review exports."""

import argparse
import asyncio
import json
from pathlib import Path

from rag_platform.config import RAGConfig
from rag_platform.evaluation import AblationExperimentSpec, run_ablation_experiments


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['dev', 'test'], default='dev')
    parser.add_argument('--embedding', choices=['local_svd', 'sentence_transformers'], default='local_svd')
    parser.add_argument('--llm-mode', choices=['local', 'openai', 'gemini'], default='local')
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2')
    args = parser.parse_args()
    root = RAGConfig().project_root
    dataset = root/'data/python_support'
    manifest = json.loads((dataset/'dataset_manifest.json').read_text())
    import hashlib
    for filename, expected in manifest['sha256'].items():
        if hashlib.sha256((dataset/filename).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Dataset hash mismatch: ' + filename)
    cfg = RAGConfig(corpus_path=dataset/'documents.jsonl', golden_qa_path=dataset/f'qa_{args.split}.jsonl',
                    reports_dir=root/'reports/python_support'/args.embedding/args.llm_mode/args.split,
                    index_dir=dataset/'index', embedding_backend=args.embedding,
                    embedding_model=args.model, llm_mode=args.llm_mode, use_cache=False)
    specs = [AblationExperimentSpec(name=name, label=label, chunking_strategy='parent_child',
                                   retrieval_mode=mode, use_reranker=rerank,
                                   description='Controlled: same documents, embedding and chunking')
             for name, label, mode, rerank in [
                 ('bm25','BM25','bm25',False), ('dense','Dense','dense',False),
                 ('hybrid','Hybrid RRF','hybrid_rrf',False),
                 ('hybrid_rerank','Hybrid RRF + feature reranker','hybrid_rrf',True)]]
    payload = await run_ablation_experiments(cfg, save_reports=True, specs=specs)
    payload['dataset_manifest'] = manifest
    payload['split'] = args.split
    payload['comparison_type'] = 'controlled_retrieval_comparison'
    (cfg.reports_dir/'ablation_results.json').write_text(json.dumps(payload, indent=2)+'\n')
    review_rows = []
    for exp in payload['experiments']:
        for row in exp['per_query']:
            review_rows.append({'architecture': exp['spec']['name'], **row})
        print(exp['spec']['label'], 'Recall@5', exp['metrics']['recall_at_5']['mean'],
              'KW coverage', exp['metrics']['keyword_coverage']['mean'],
              'P95 ms', exp['latency_distribution_ms']['p95'])
    review_path = cfg.reports_dir/'pending_human_review.jsonl'
    review_path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in review_rows))
    print('Pending human review:', review_path)
    if args.llm_mode != 'local' and any(row['degraded_mode'] for row in review_rows):
        raise SystemExit('Provider fallback detected. Results must not be counted as successful real-model evaluation.')


if __name__ == '__main__':
    asyncio.run(main())
