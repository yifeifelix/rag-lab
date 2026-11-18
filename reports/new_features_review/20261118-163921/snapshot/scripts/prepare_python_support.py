"""Build a real, offline corpus from the installed CPython asyncio docstrings.

QA labels are agent-authored and require human review. They are not a validated
independent benchmark. Existing files are frozen: use a new directory for revisions.
"""

import asyncio
import hashlib
import inspect
import json
from pathlib import Path
import platform
import sys
import shutil

from rag_platform.config import RAGConfig


def prepare():
    root = RAGConfig().project_root / 'data/python_support'
    root.mkdir(parents=True, exist_ok=True)
    docs = []
    names = ['gather', 'wait_for', 'wait', 'create_task', 'shield', 'to_thread', 'TaskGroup', 'Semaphore', 'Queue']
    for name in names:
        obj = getattr(asyncio, name)
        source = Path(inspect.getsourcefile(obj))
        line = inspect.getsourcelines(obj)[1]
        text = f'# Python asyncio.{name}\n\n' + inspect.getdoc(obj)
        docs.append({'doc_id': f'PY-{name.upper().replace("_", "-")}',
                     'title': f'Python asyncio.{name}', 'domain': 'python_asyncio',
                     'version': platform.python_version(), 'text': text,
                     'source': f'https://docs.python.org/3.12/library/asyncio-{("sync" if name == "Semaphore" else "queue" if name == "Queue" else "task")}.html#asyncio.{name}',
                     'license': 'Python Software Foundation License Version 2',
                     'provenance': {'kind': 'installed_cpython_docstring', 'local_source': str(source),
                                    'source_line': line, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                                    'docstring_sha256': hashlib.sha256(text.encode()).hexdigest()}})
    # Different questions and IDs in each split. Same author/corpus: human independence is not claimed.
    dev = [
        ('gather', 'Does asyncio.gather return successful results in completion order or input order?', ['original sequence']),
        ('wait_for', 'What exception does asyncio.wait_for raise when its timeout expires?', ['TimeoutError']),
        ('wait', 'Which two sets does asyncio.wait return?', ['done', 'pending']),
        ('create_task', 'What does asyncio.create_task return after scheduling a coroutine?', ['Task']),
        ('shield', 'What exception can the caller still receive while awaiting asyncio.shield?', ['CancelledError']),
        ('to_thread', 'Does asyncio.to_thread propagate contextvars.Context to the worker thread?', ['contextvars', 'propagated']),
        ('TaskGroup', 'When does an asyncio.TaskGroup await all its tasks?', ['context manager exits']),
        ('Semaphore', 'How does release affect the internal counter of asyncio.Semaphore?', ['incremented']),
        ('Queue', 'How can a consumer inspect the current size of an asyncio.Queue?', ['qsize']),
    ]
    test = [
        ('gather', 'How does return_exceptions=True change asyncio.gather exception handling?', ['exceptions', 'result list']),
        ('gather', 'What happens to unfinished children when the outer asyncio.gather future is cancelled?', ['cancelled']),
        ('wait_for', 'How can I prevent asyncio.wait_for from cancelling a task on timeout?', ['shield']),
        ('wait_for', 'What if the coroutine being awaited by asyncio.wait_for suppresses cancellation and returns a value?', ['value', 'returned']),
        ('wait', 'What happens to unfinished futures when asyncio.wait reaches its timeout?', ['pending', 'does not raise TimeoutError']),
        ('create_task', 'Which function schedules a coroutine object and returns an asyncio Task?', ['create_task']),
        ('shield', 'Why should callers retain a reference to a task passed to asyncio.shield?', ['weak references', 'garbage collected']),
        ('to_thread', 'How are positional and keyword arguments supplied to asyncio.to_thread passed to the target?', ['directly passed']),
        ('TaskGroup', 'What does asyncio.TaskGroup do when a task raises an exception other than CancelledError?', ['cancel', 'ExceptionGroup']),
        ('Semaphore', 'What occurs if asyncio.Semaphore is created with a negative initial value?', ['ValueError']),
        ('Queue', 'When does await put block for an asyncio.Queue with positive maxsize?', ['maxsize', 'removed']),
        (None, 'What is the capital city of the fictional kingdom of Zyxoria?', []),
        (None, 'Which winning lottery numbers will be drawn tomorrow?', []),
    ]
    def items(rows, split):
        return [{'question_id': f'PY-{split.upper()}-{i:02}',
                 'category': 'conceptual_semantic' if name else 'unanswerable_guardrail',
                 'question': question, 'relevant_doc_ids': [f'PY-{name.upper().replace("_", "-")}'] if name else [],
                 'expected_keywords': keywords, 'paraphrase_variants': [],
                 'split': split, 'label_status': 'agent_authored_pending_human_review'}
                for i, (name, question, keywords) in enumerate(rows, 1)]
    artifacts = {'documents.jsonl': docs, 'qa_dev.jsonl': items(dev, 'dev'), 'qa_test.jsonl': items(test, 'test')}
    hashes = {}
    for filename, rows in artifacts.items():
        content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
        path = root / filename
        if path.exists() and path.read_text() != content:
            raise RuntimeError(f'{filename} is frozen; use a new dataset version')
        path.write_text(content)
        hashes[filename] = hashlib.sha256(content.encode()).hexdigest()
    license_path = Path(sys.base_prefix) / 'LICENSE.txt'
    if license_path.exists():
        (root/'PYTHON_LICENSE.txt').write_text(license_path.read_text())
    manifest = {'python_version': platform.python_version(), 'corpus_kind': 'real_cpython_docstrings',
                'documents': len(docs), 'dev_questions': len(dev), 'test_questions': len(test),
                'human_review_status': 'pending', 'independent_test_author': False,
                'sha256': hashes,
                'limitations': ['Small technical domain', 'Not a multilingual benchmark',
                                'Current runtime docstrings may lag online documentation',
                                'Test split is frozen but not blind to the implementation agent']}
    (root/'dataset_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    # A mutable demo workspace must not modify the frozen evaluation corpus.
    workspace = root.parent/'workspaces/python_support'
    workspace.mkdir(parents=True, exist_ok=True)
    for name in ('documents.jsonl', 'qa_dev.jsonl'):
        if not (workspace/name).exists():
            shutil.copyfile(root/name, workspace/name)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    prepare()
