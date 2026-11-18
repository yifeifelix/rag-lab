"""Explicit data roots make containers/CI portable while preserving the local default."""
import os
from pathlib import Path
import subprocess
import sys


def test_explicit_data_root_relocates_runtime(tmp_path):
    root = tmp_path / 'container' / 'app'
    root.mkdir(parents=True)
    env = dict(os.environ, RAG_PROJECT_ROOT=str(root), RAG_DATA_ROOT=str(root.parent),
               HF_HOME='/outside-approved-root/models', TMPDIR='/outside-approved-root/tmp')
    code = """
from pathlib import Path
import os
from rag_platform.runtime import configure_runtime
paths = configure_runtime()
root = Path(os.environ['RAG_PROJECT_ROOT'])
assert all(Path(value).is_relative_to(root) for value in paths.values())
assert all(Path(value).is_dir() for value in paths.values())
"""
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_project_outside_data_root_is_rejected(tmp_path):
    env = dict(os.environ, RAG_PROJECT_ROOT=str(tmp_path / 'app'),
               RAG_DATA_ROOT=str(tmp_path / 'different-volume'))
    result = subprocess.run([sys.executable, '-c',
                             'from rag_platform.runtime import configure_runtime; configure_runtime()'],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'must reside under RAG_DATA_ROOT' in result.stderr
