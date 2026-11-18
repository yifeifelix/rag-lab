"""Loading the course corpora and question sets."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path


def data_dir() -> Path:
    """Folder holding the corpora (override with RAGLAB_DATA)."""
    env = os.getenv("RAGLAB_DATA")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data"


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    source: str = ""
    metadata: dict = field(default_factory=dict)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_documents(corpus: str = "python_docs") -> list[Document]:
    """Load a corpus: ``"python_docs"`` (real) or ``"policies"`` (fictional traps)."""
    rows = _read_jsonl(data_dir() / corpus / "documents.jsonl")
    docs = []
    for row in rows:
        meta = {k: v for k, v in row.items() if k not in ("doc_id", "title", "text", "source")}
        docs.append(Document(row["doc_id"], row["title"], row["text"], row.get("source", ""), meta))
    return docs


def load_qa(corpus: str = "python_docs") -> list[dict]:
    """Load evaluation questions. Each has ``question`` and ``relevant_doc_ids``.

    An empty ``relevant_doc_ids`` list means the corpus cannot answer it.
    """
    return _read_jsonl(data_dir() / corpus / "qa.jsonl")
