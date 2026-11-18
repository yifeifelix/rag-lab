"""Keep tests away from live documents, credentials, paid models and reports."""

from pathlib import Path
import shutil

import pytest


@pytest.fixture(autouse=True)
def isolated_test_environment(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv('RAG_LLM_MODE', 'local')
    monkeypatch.setenv('RAG_EMBEDDING_BACKEND', 'local_svd')
    monkeypatch.setenv('RAG_ADMIN_TOKEN', '')
    monkeypatch.setenv('RAG_CORPUS_PATH', str(tmp_path/'fixture/corpus.jsonl'))
    monkeypatch.setenv('RAG_QA_PATH', str(tmp_path/'fixture/qa.jsonl'))
    monkeypatch.setenv('RAG_INDEX_DIR', str(tmp_path/'fixture/index'))
    monkeypatch.setenv('RAG_BOOTSTRAP_SYNTHETIC', 'true')
    reports = tmp_path/'reports'
    reports.mkdir()
    for filename in ('ablation_results.json', 'benchmark_10k.json'):
        path = root/'reports'/filename
        if path.exists():
            shutil.copyfile(path, reports/filename)
    monkeypatch.setenv('RAG_REPORTS_DIR', str(reports))
    from rag_platform.config import RAGConfig
    from rag_platform.data_gen import generate_datasets
    generate_datasets(RAGConfig(), force=False)
