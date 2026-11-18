"""Explicit workspace initialization; imports never create files."""
import os
from pathlib import Path
import tempfile


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def initialize_workspace(config):
    seed = config.project_root / 'data/learning_v2/documents.jsonl'
    if not config.corpus_path.exists():
        if config.bootstrap_synthetic and seed.exists():
            atomic_text(config.corpus_path, seed.read_text())
        elif config.bootstrap_synthetic:
            from .data_gen import generate_datasets
            generate_datasets(config, force=False)
        else:
            atomic_text(config.corpus_path, '')
    if not config.golden_qa_path.exists():
        atomic_text(config.golden_qa_path, '')
