"""Development study and a separately frozen final-test entry point."""
import argparse
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json
import uuid
from pathlib import Path

from rag_platform.bootstrap import atomic_text, initialize_workspace
from rag_platform.config import RAGConfig
from rag_platform.evaluation import evaluate_pipeline
from rag_platform.pipeline import EnterpriseRAGPipeline
from rag_platform.provenance import safe_config
from rag_platform.runtime import configure_runtime
from rag_platform.study import RRFGridSearch, calibrate, config_signature, load_dev, sha


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['tune', 'calibrate', 'compare', 'freeze', 'final-test'])
    parser.add_argument('--embedding', choices=['local_svd', 'sentence_transformers'])
    parser.add_argument('--reranker', choices=['features', 'cross_encoder'])
    parser.add_argument('--include-neural', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--freeze', type=Path)
    args = parser.parse_args()
    configure_runtime()
    cfg = RAGConfig()
    initialize_workspace(cfg)
    cfg = replace(cfg, golden_qa_path=cfg.dev_qa_path, use_cache=False,
                  calibration_path=None if args.action in ('calibrate', 'tune', 'compare') else cfg.calibration_path,
                  embedding_backend=args.embedding or cfg.embedding_backend,
                  reranker_backend=args.reranker or cfg.reranker_backend)
    pipeline = EnterpriseRAGPipeline(cfg)
    cfg = pipeline.config
    root = cfg.reports_dir / 'learning_v2'
    root.mkdir(parents=True, exist_ok=True)
    try:
        if args.action == 'tune':
            result = RRFGridSearch.tune(pipeline, load_dev(cfg))
        elif args.action == 'calibrate':
            result = await calibrate(pipeline, load_dev(cfg))
        elif args.action == 'compare':
            rows = load_dev(cfg)
            comparisons = []
            variants = [('bm25', False, 'features'), ('dense', False, 'features'),
                        ('hybrid_rrf', False, 'features'), ('hybrid_rrf', True, 'features')]
            if args.include_neural:
                variants.append(('hybrid_rrf', True, 'cross_encoder'))
            for mode, rerank, backend in variants:
                variant = replace(cfg, retrieval_mode=mode, use_reranker=rerank, reranker_backend=backend)
                candidate = EnterpriseRAGPipeline(variant, pipeline.documents, previous=pipeline)
                try:
                    comparison = await evaluate_pipeline(candidate, rows)
                    comparisons.append({'retrieval_mode': mode, 'use_reranker': rerank,
                                        'reranker_backend': backend, **comparison})
                finally:
                    await candidate.aclose()
            result = {'split': 'dev', 'dev_sha256': sha(cfg.dev_qa_path), 'comparisons': comparisons,
                      'label_status': 'agent_authored_pending_human_review'}
        elif args.action == 'freeze':
            manifest = json.loads((cfg.dev_qa_path.parent/'dataset_manifest.json').read_text())
            result = {'config_signature': config_signature(cfg), 'corpus_revision': pipeline.corpus_revision,
                      'runner_sha256': sha(__file__),
                      'dev_sha256': sha(cfg.dev_qa_path), 'test_sha256': manifest['sha256'][cfg.test_qa_path.name],
                      'calibration_sha256': sha(cfg.calibration_path) if cfg.calibration_path else None,
                      'configuration': safe_config(cfg),
                      'time': datetime.now(timezone.utc).isoformat(),
                      'blind_test': False, 'notes': 'Freeze precedes test reading; agent-authored test is not independent.'}
        else:
            if not args.freeze:
                raise SystemExit('final-test requires --freeze PATH from the freeze command')
            frozen = json.loads(args.freeze.read_text())
            if (frozen['config_signature'] != config_signature(cfg)
                    or frozen['runner_sha256'] != sha(__file__)
                    or frozen['corpus_revision'] != pipeline.corpus_revision
                    or frozen['test_sha256'] != sha(cfg.test_qa_path)
                    or frozen['calibration_sha256'] != (sha(cfg.calibration_path) if cfg.calibration_path else None)):
                raise SystemExit('Frozen code, corpus, model, calibration or test changed; create a new freeze record')
            rows = [json.loads(line) for line in cfg.test_qa_path.read_text().splitlines() if line.strip()]
            if not rows or any(row.get('split') != 'test' for row in rows):
                raise SystemExit('Final test requires explicitly labelled test rows')
            result = {'split': 'test', 'freeze_sha256': sha(args.freeze), 'frozen_selection': frozen,
                      **await evaluate_pipeline(pipeline, rows)}
        run_id = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
        output = args.output or root/run_id/(args.action + '.json')
        atomic_text(output, json.dumps(result, indent=2, ensure_ascii=False) + '\n')
        print('Saved', output)
        if args.action == 'calibrate':
            print('OOF Brier:', result['oof_brier_score'], 'operating point:', result['oof_selected_operating_point'])
    finally:
        await pipeline.aclose()


if __name__ == '__main__':
    asyncio.run(main())
