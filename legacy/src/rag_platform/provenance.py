"""Reproducibility metadata without secrets or raw runtime environment values."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform
from dataclasses import asdict
import json


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_config(config):
    values = asdict(config)
    for key in ('admin_token', 'tester_token', 'qdrant_url'):
        values.pop(key, None)
    return json.loads(json.dumps(values, default=str))


def experiment_provenance(config):
    source_dir = Path(__file__).parent
    return {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'configuration': safe_config(config),
        'python': platform.python_version(), 'platform': platform.platform(),
        'dependencies': {p: version(p) for p in ('numpy', 'scipy', 'scikit-learn', 'fastapi', 'httpx')},
        'source_sha256': {p.name: sha256_file(p) for p in sorted(source_dir.glob('*.py'))},
        'corpus_sha256': sha256_file(config.corpus_path),
        'qa_sha256': sha256_file(config.golden_qa_path),
        'llm_mode': config.llm_mode, 'embedding_backend': config.embedding_backend,
        'embedding_revision': config.embedding_revision,
        'reranker_backend': config.reranker_backend,
        'reranker_model': config.reranker_model if config.reranker_backend == 'cross_encoder' else None,
        'reranker_revision': config.reranker_revision,
        'vector_backend': config.vector_backend,
        'embedding_model': config.embedding_model if config.embedding_backend != 'local_svd' else None,
        'model_api_validated': False, 'bootstrap_seed': 42,
    }
