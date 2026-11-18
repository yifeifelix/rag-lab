"""Development-only tuning and group cross-fitted answerability calibration."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from .compute import run_cpu
from .security import redact_pii


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_signature(config):
    fields = ('chunking_strategy', 'fixed_chunk_size', 'fixed_overlap', 'recursive_max_chars',
              'parent_max_chars', 'child_max_chars', 'semantic_breakpoint_percentile',
              'embedding_backend', 'embedding_model', 'embedding_revision', 'embedding_dim', 'vector_backend',
              'reranker_backend', 'reranker_model', 'reranker_revision', 'dense_weight',
              'sparse_weight', 'graph_weight', 'rrf_k', 'top_k_candidates', 'top_k_final',
              'retrieval_mode', 'use_reranker', 'enable_crag', 'enable_graph_rag',
              'enable_query_expansion', 'enable_contextual_compression', 'enable_contextual_headers',
              'feature_rank_weights', 'feature_confidence_weights', 'bm25_scale', 'query_vocab_floor',
              'dense_confidence_weights', 'sparse_confidence_weights', 'hybrid_confidence_weights', 'grounding_min_terms',
              'crag_ambiguous_threshold', 'crag_vocab_threshold')
    body = {key: getattr(config, key) for key in fields}
    body['min_confidence_threshold'] = config.min_confidence_threshold
    body['llm_mode'] = config.llm_mode
    body['use_calibration'] = config.use_calibration
    import os
    body['generation_model'] = (os.getenv('OPENAI_MODEL', 'gpt-4o-mini') if config.llm_mode == 'openai'
                                else os.getenv('GEMINI_MODEL', '') if config.llm_mode == 'gemini' else None)
    body['rules_hash'] = sha(config.expansion_rules_path) if config.expansion_rules_path else None
    body['source_hashes'] = {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')}
    from importlib.metadata import version
    packages = ['numpy', 'scipy', 'scikit-learn']
    if config.embedding_backend == 'sentence_transformers' or config.reranker_backend == 'cross_encoder':
        packages += ['sentence-transformers', 'torch', 'transformers']
    body['dependencies'] = {package: version(package) for package in packages}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def load_dev(config):
    path = Path(config.dev_qa_path)
    if path.resolve() == Path(config.test_qa_path).resolve():
        raise ValueError('Development and test paths must differ')
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or any(row.get('split') != 'dev' or not row.get('group_id') for row in rows):
        raise ValueError('Tuning/calibration requires explicitly labelled, grouped development rows')
    if len({row['question_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate development question IDs')
    manifest_path = path.parent / 'dataset_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('sha256', {}).get(path.name) != sha(path):
            raise ValueError('Development data hash does not match the frozen manifest')
        groups = manifest.get('groups', {})
        if set(groups.get('dev', [])) & set(groups.get('test', [])):
            raise ValueError('Development/test groups overlap')
        if groups.get('dev') and {row['group_id'] for row in rows} != set(groups['dev']):
            raise ValueError('Development groups do not match the manifest')
        question_hashes = manifest.get('question_hashes', {})
        if set(question_hashes.get('dev', [])) & set(question_hashes.get('test', [])):
            raise ValueError('Development/test questions overlap')
    return rows


def retrieval_observation(pipeline, item, weights=None):
    cfg = pipeline.config
    query = redact_pii(item['question'])[0]
    selected, status, _, score, _, _ = pipeline.retrieve_context_sync(
        query, retrieval_mode=cfg.retrieval_mode, use_reranker=cfg.use_reranker,
        top_k=cfg.top_k_final, weights=weights)
    selected, evidence, _ = pipeline.prepare_evidence_sync(query, selected)
    relevant = set(item.get('relevant_doc_ids', []))
    got = list(dict.fromkeys(ev.doc_id for ev in evidence))
    spans = item.get('supporting_spans', {})
    support = bool(relevant) and relevant <= set(got)
    for doc_id, quotations in spans.items():
        context = '\n'.join(ev.parent_text for ev in evidence if ev.doc_id == doc_id)
        normalized = ' '.join(context.lower().split())
        support = support and all(' '.join(quotation.lower().split()) in normalized for quotation in quotations)
    if status == 'INCORRECT':
        score, support = 0.0, False
    accepted = bool(selected) and status != 'INCORRECT' and score >= cfg.min_confidence_threshold
    return {'score': score, 'supported': int(support), 'group_id': item['group_id'],
            'question_id': item['question_id'], 'ranked_doc_ids': got if accepted else [],
            'router_status': status}


class RRFGridSearch:
    """Five explicit RRF trials on development data through the serving path."""

    @staticmethod
    def tune(pipeline, rows):
        if not rows or any(row.get('split') != 'dev' for row in rows):
            raise ValueError('Only development questions may be used for tuning')
        answerable = [row for row in rows if row.get('relevant_doc_ids')]
        if not answerable:
            raise ValueError('No answerable development questions')
        known = set(pipeline.document_by_id)
        if any(set(row['relevant_doc_ids']) - known for row in answerable):
            raise ValueError('Development questions do not match the active corpus')
        graph = pipeline.config.enable_graph_rag
        space = [(0.55, 0.45, 0.0, 60), (0.70, 0.30, 0.0, 60),
                 (0.30, 0.70, 0.0, 60), (0.55, 0.45, 0.0, 40),
                 (0.45, 0.35, 0.20, 60) if graph else (0.55, 0.45, 0.0, 80)]
        trials = []
        for dw, sw, gw, rk in space:
            recalls, mrrs, ndcgs = [], [], []
            for item in answerable:
                got = retrieval_observation(pipeline, item, (dw, sw, gw, rk))['ranked_doc_ids'][:5]
                relevant = set(item['relevant_doc_ids'])
                recalls.append(len(relevant & set(got)) / len(relevant))
                mrrs.append(next((1 / rank for rank, doc in enumerate(got, 1) if doc in relevant), 0))
                dcg = sum(1 / math.log2(rank + 1) for rank, doc in enumerate(got, 1) if doc in relevant)
                ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(5, len(relevant)) + 1))
                ndcgs.append(dcg / ideal)
            trials.append(dict(dense_weight=dw, sparse_weight=sw, graph_weight=gw, rrf_k=rk,
                               recall_at_5=float(np.mean(recalls)), mrr=float(np.mean(mrrs)),
                               ndcg_at_5=float(np.mean(ndcgs))))
        trials.sort(key=lambda row: (row['ndcg_at_5'], row['mrr'], row['recall_at_5']), reverse=True)
        return {'method': 'RRFGridSearch', 'best_config': trials[0], 'all_trials': trials,
                'split': 'dev', 'num_queries_evaluated': len(answerable), 'applied': False,
                'requires_recalibration': True,
                'objective': 'development retrieval NDCG with the raw relevance gate; not final answer correctness',
                'dev_sha256': sha(pipeline.config.dev_qa_path),
                'corpus_revision': pipeline.corpus_revision, 'config_signature': config_signature(pipeline.config)}


class CalibrationMismatch(ValueError):
    pass


@dataclass
class ConfidenceCalibration:
    x: list[float]
    y: list[float]
    threshold: float
    signature: str
    corpus_revision: str

    def predict(self, value):
        return float(np.interp(value, self.x, self.y))

    @classmethod
    def load(cls, path, config, corpus_revision=None):
        data = json.loads(Path(path).read_text())
        if data.get('split') != 'dev' or data['model']['signature'] != config_signature(config):
            raise CalibrationMismatch('Calibration configuration/source mismatch')
        model = cls(**data['model'])
        if (not model.x or len(model.x) != len(model.y) or model.x != sorted(model.x)
                or not np.isfinite(model.x + model.y + [model.threshold]).all()
                or not all(0 <= value <= 1 for value in model.y)
                or not 0 <= model.threshold <= 1.000001):
            raise CalibrationMismatch('Invalid calibration model values')
        if corpus_revision and model.corpus_revision != corpus_revision:
            raise CalibrationMismatch('Corpus changed; recalibration required')
        return model


async def calibrate(pipeline, rows):
    if not rows or any(row.get('split') != 'dev' for row in rows):
        raise ValueError('Calibration may only use development rows')
    observations = [await run_cpu(retrieval_observation, pipeline, row, workers=pipeline.config.cpu_workers)
                    for row in rows]
    x = np.array([row['score'] for row in observations])
    y = np.array([row['supported'] for row in observations])
    groups = np.array([row['group_id'] for row in observations])
    if len(set(groups)) < 3 or len(set(y)) != 2:
        raise ValueError('Calibration needs at least three groups and both sufficient/insufficient support labels')
    oof = np.zeros_like(x)
    for train, validation in GroupKFold(n_splits=min(5, len(set(groups)))).split(x, y, groups):
        if len(set(y[train])) < 2:
            raise ValueError('Each calibration training fold needs both support labels')
        estimator = IsotonicRegression(out_of_bounds='clip').fit(x[train], y[train])
        oof[validation] = estimator.predict(x[validation])
    thresholds = sorted(set([0.0, 1.000001, *oof.tolist()]))
    tradeoffs = []
    for threshold in thresholds:
        accept = oof >= threshold
        false_answer = int(np.sum(accept & (y == 0)))
        false_refusal = int(np.sum(~accept & (y == 1)))
        tradeoffs.append({'threshold': threshold, 'cost': 4 * false_answer + false_refusal,
                         'unsupported_accepts': false_answer, 'false_refusals': false_refusal,
                         'coverage': float(np.mean(accept))})
    best = min(tradeoffs, key=lambda row: (row['cost'], -row['threshold']))
    final = IsotonicRegression(out_of_bounds='clip').fit(x, y)
    model = ConfidenceCalibration(final.X_thresholds_.tolist(), final.y_thresholds_.tolist(),
                                  best['threshold'], config_signature(pipeline.config), pipeline.corpus_revision)
    return {'method': 'isotonic_group_cross_fit', 'split': 'dev', 'model': asdict(model),
            'target': 'all_required_retrieved_evidence_present; not factual answer correctness',
            'dev_sha256': sha(pipeline.config.dev_qa_path), 'questions': len(rows),
            'groups': len(set(groups)), 'oof_brier_score': float(np.mean((oof - y) ** 2)),
            'oof_selected_operating_point': best, 'threshold_tradeoffs': tradeoffs,
            'label_status': 'agent_authored_pending_independent_human_review',
            'observations': observations}
