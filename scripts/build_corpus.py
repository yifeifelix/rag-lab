"""Build the course corpora.

1. ``data/python_docs/documents.jsonl`` - real documentation pulled from the
   docstrings of the Python 3.12 standard library (PSF License).
2. ``data/policies/documents.jsonl`` + ``qa.jsonl`` - a small fictional policy
   set (MIT) that contains deliberate traps: superseded versions, look-alike
   regions and questions with no answer.

Run with the project's Python 3.12 so the output is reproducible:

    .venv/bin/python scripts/build_corpus.py
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Modules picked to give a mix of short and long, overlapping and distinct topics.
MODULES = [
    "json", "csv", "pathlib", "datetime", "collections", "itertools", "functools",
    "re", "logging", "argparse", "sqlite3", "subprocess", "threading", "queue",
    "heapq", "bisect", "statistics", "random", "urllib.parse", "shutil", "tempfile",
    "textwrap", "string", "enum", "dataclasses", "contextlib", "copy", "pickle",
    "hashlib", "secrets", "base64", "zipfile", "gzip", "time", "calendar",
    "asyncio.tasks", "asyncio.queues", "asyncio.locks", "concurrent.futures",
    "unittest.mock", "typing", "operator", "math", "fractions", "decimal",
]

MAX_MEMBERS = 70
MAX_MEMBER_CHARS = 1500


def _clean(doc: str | None) -> str:
    return inspect.cleandoc(doc).strip() if isinstance(doc, str) else ""


def module_document(name: str) -> dict | None:
    module = importlib.import_module(name)
    intro = _clean(module.__doc__)
    public = getattr(module, "__all__", None) or [n for n in dir(module) if not n.startswith("_")]
    sections = []
    for member_name in sorted(public):
        member = getattr(module, member_name, None)
        # Only document classes and functions; data members inherit their type's docstring.
        if not (inspect.isclass(member) or inspect.isroutine(member)):
            continue
        # Skip names re-exported from other modules (keeps documents distinct).
        owner = getattr(member, "__module__", None) or name
        if not (owner.startswith(name) or owner == f"_{name}" or name.startswith("asyncio")):
            continue
        doc = _clean(getattr(member, "__doc__", None))
        if len(doc) < 20:
            continue
        kind = "class" if inspect.isclass(member) else "function"
        try:
            signature = str(inspect.signature(member)) if callable(member) else ""
        except (TypeError, ValueError):
            signature = ""
        sections.append(f"## {name}.{member_name}{signature}\n\n*{kind}*\n\n{doc[:MAX_MEMBER_CHARS]}")
        if len(sections) >= MAX_MEMBERS:
            break
    if not intro and not sections:
        return None
    text = f"# The `{name}` module\n\n{intro}\n\n" + "\n\n".join(sections)
    page = name.split(".")[0] if not name.startswith(("asyncio", "concurrent", "unittest", "urllib")) else name
    return {
        "doc_id": f"py.{name}",
        "title": f"Python standard library: {name}",
        "source": f"https://docs.python.org/3.12/library/{page}.html",
        "license": "PSF-2.0",
        "python_version": sys.version.split()[0],
        "text": text.strip() + "\n",
    }


POLICY_REGIONS = {"EU": 17, "US": 25, "APAC": 40}
POLICY_RETENTION = {"EU": 30, "US": 90, "APAC": 60}


def policy_documents() -> list[dict]:
    docs = []
    for region, minutes in POLICY_REGIONS.items():
        docs.append({
            "doc_id": f"cedar.{region.lower()}.restore.v2",
            "title": f"Cedar {region} restore policy (2026, current)",
            "text": (f"# Cedar {region} restore policy\nStatus: CURRENT. Effective version: 2026.\n"
                     f"The Cedar {region} restore recovery target is {minutes} minutes.\n"
                     f"This policy applies only to the {region} region and the restore operation."),
        })
        docs.append({
            "doc_id": f"cedar.{region.lower()}.restore.v1",
            "title": f"Cedar {region} restore policy (2024, superseded)",
            "text": (f"# Cedar {region} restore policy\nStatus: SUPERSEDED by the 2026 version.\n"
                     f"The Cedar {region} restore recovery target was {minutes * 3} minutes."),
        })
        docs.append({
            "doc_id": f"cedar.{region.lower()}.export.v2",
            "title": f"Cedar {region} export retention policy (2026, current)",
            "text": (f"# Cedar {region} export policy\nStatus: CURRENT. Effective version: 2026.\n"
                     f"Exported Cedar {region} data is retained for {POLICY_RETENTION[region]} days, "
                     f"then deleted automatically."),
        })
    for doc in docs:
        doc.update(source=f"fixture://policies/{doc['doc_id']}", license="MIT")
    return docs


def policy_questions() -> list[dict]:
    qa = []
    for region, minutes in POLICY_REGIONS.items():
        qa.append({"question": f"What is the current restore recovery target for Cedar {region}?",
                   "answer": f"{minutes} minutes", "relevant_doc_ids": [f"cedar.{region.lower()}.restore.v2"],
                   "category": "version_trap"})
        qa.append({"question": f"How long is exported Cedar {region} data kept?",
                   "answer": f"{POLICY_RETENTION[region]} days",
                   "relevant_doc_ids": [f"cedar.{region.lower()}.export.v2"], "category": "region_trap"})
    qa.append({"question": "Under the 2026 Cedar EU policies, what are the restore target and the export retention period?",
               "answer": "17 minutes and 30 days",
               "relevant_doc_ids": ["cedar.eu.restore.v2", "cedar.eu.export.v2"], "category": "multi_hop"})
    for question in ("What is the Cedar contract renewal fee?", "Does Cedar offer a refund guarantee?",
                     "How many on-call engineers does Cedar EU require?"):
        qa.append({"question": question, "answer": None, "relevant_doc_ids": [], "category": "unanswerable"})
    for i, item in enumerate(qa, 1):
        item["qid"] = f"POL-{i:02d}"
    return qa


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main() -> None:
    docs = [d for d in (module_document(m) for m in MODULES) if d]
    write_jsonl(ROOT / "data/python_docs/documents.jsonl", docs)
    write_jsonl(ROOT / "data/policies/documents.jsonl", policy_documents())
    write_jsonl(ROOT / "data/policies/qa.jsonl", policy_questions())
    sizes = sorted(len(d["text"]) for d in docs)
    print(f"python_docs: {len(docs)} documents, chars min/median/max = "
          f"{sizes[0]}/{sizes[len(sizes) // 2]}/{sizes[-1]}")


if __name__ == "__main__":
    main()
