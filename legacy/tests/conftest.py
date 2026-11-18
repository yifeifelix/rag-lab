"""Keep tests away from live documents, credentials, paid models and reports."""

from pathlib import Path
import json
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
    monkeypatch.setenv('RAG_DEV_QA_PATH', str(tmp_path/'fixture/qa_dev.jsonl'))
    monkeypatch.setenv('RAG_TEST_QA_PATH', str(tmp_path/'fixture/qa_test.jsonl'))
    monkeypatch.setenv('RAG_TESTER_TOKEN', '')
    monkeypatch.delenv('RAG_CALIBRATION_PATH', raising=False)
    for flag in ('ROUTER', 'GRAPH', 'EXPANSION', 'COMPRESSION', 'HEADERS'):
        monkeypatch.setenv('RAG_ENABLE_' + flag, 'false')
    reports = tmp_path/'reports'
    reports.mkdir()
    for filename in ('ablation_results.json', 'benchmark_10k.json'):
        path = root/'reports'/filename
        if path.exists():
            shutil.copyfile(path, reports/filename)
    monkeypatch.setenv('RAG_REPORTS_DIR', str(reports))
    from rag_platform.config import RAGConfig
    from rag_platform.data_gen import generate_datasets
    cfg = RAGConfig()
    generate_datasets(cfg, force=False)
    rows = [json.loads(line) for line in cfg.golden_qa_path.read_text().splitlines()]
    cfg.dev_qa_path.write_text(''.join(json.dumps({**row, 'split':'dev', 'group_id':row['question_id']})+'\n' for row in rows))
    cfg.test_qa_path.write_text('')
