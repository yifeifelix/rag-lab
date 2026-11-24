"""A small text-classification dataset for Lesson 11, built from our own library.

Every section of a Python documentation page gets the topic of its page. The
test set uses *whole pages the model never saw during training*, so a model
can't cheat by memorising page-specific names like ``csv.reader``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .chunking import chunk_documents
from .data import load_documents

TOPICS: dict[str, list[str]] = {
    "files & data formats": ["json", "csv", "pathlib", "pickle", "shutil", "tempfile", "zipfile", "gzip",
                             "base64", "sqlite3"],
    "numbers & maths": ["math", "statistics", "random", "fractions", "decimal"],
    "running things at the same time": ["threading", "queue", "asyncio.tasks", "asyncio.queues",
                                        "asyncio.locks", "subprocess", "concurrent.futures"],
    "text": ["re", "string", "textwrap"],
}
#: Pages held back for the final test (one or two per topic).
TEST_PAGES = {"py.csv", "py.gzip", "py.statistics", "py.fractions", "py.asyncio.queues",
              "py.subprocess", "py.textwrap"}


@dataclass
class Example:
    text: str
    label: str
    doc_id: str


def topic_examples() -> list[Example]:
    """Every documentation section (longer than 80 characters) with its topic label."""
    page_topic = {f"py.{m}": topic for topic, modules in TOPICS.items() for m in modules}
    sections = chunk_documents(load_documents("python_docs"), method="section")
    return [Example(s.text, page_topic[s.doc_id], s.doc_id)
            for s in sections if s.doc_id in page_topic and len(s.text) > 80]


def topic_split() -> tuple[list[Example], list[Example]]:
    """(train, test) - the test set contains only pages never seen in training."""
    examples = topic_examples()
    return ([e for e in examples if e.doc_id not in TEST_PAGES],
            [e for e in examples if e.doc_id in TEST_PAGES])
