"""Single-owner document lifecycle with atomic corpus commits and index rebuilding."""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading

from .data_gen import generate_datasets, load_corpus_documents
from .pipeline import EnterpriseRAGPipeline, _redact_pii


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class RevisionConflict(ValueError):
    pass


class KnowledgeBase:
    """Each request refreshes on corpus change; in-flight requests keep their snapshot.

    This is a local filesystem implementation, not a distributed database. Writers
    coordinate through a POSIX lock and publish a new pipeline only after building it.
    """

    def __init__(self, config):
        self.config = config
        self._lock = threading.RLock()
        self.lock_path = config.corpus_path.with_suffix('.lock')
        self.events_path = config.corpus_path.parent / 'document_events.jsonl'
        config.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        with self._file_lock():
            if not config.corpus_path.exists():
                if config.bootstrap_synthetic:
                    generate_datasets(config, force=False)
                else:
                    atomic_write(config.corpus_path, '')
            if not config.golden_qa_path.exists():
                atomic_write(config.golden_qa_path, '')
            self._pipeline = EnterpriseRAGPipeline(config, load_corpus_documents(config.corpus_path))
            self._signature = self._stat()

    @contextmanager
    def _file_lock(self):
        with self.lock_path.open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _stat(self):
        stat = self.config.corpus_path.stat()
        return stat.st_mtime_ns, stat.st_size, stat.st_ino

    def current(self):
        with self._lock:
            if self._stat() != self._signature:
                with self._file_lock():
                    self._pipeline = EnterpriseRAGPipeline(
                        self.config, load_corpus_documents(self.config.corpus_path))
                    self._signature = self._stat()
            return self._pipeline

    def mutate(self, document=None, delete_id=None, expected_revision=None):
        with self._lock, self._file_lock():
            docs = load_corpus_documents(self.config.corpus_path)
            from .index_store import fingerprint
            previous = fingerprint(docs, self.config)
            if expected_revision and expected_revision != previous:
                raise RevisionConflict('Corpus changed; reload documents before retrying')
            doc_id = delete_id or document['doc_id']
            exists = any(d['doc_id'] == doc_id for d in docs)
            if delete_id and not exists:
                raise KeyError(doc_id)
            updated = [d for d in docs if d['doc_id'] != doc_id]
            redacted = False
            if document is not None:
                document = dict(document)
                for key, value in document.items():
                    if isinstance(value, str) and key != 'doc_id':
                        document[key], changed = _redact_pii(value)
                        redacted |= changed
                document['updated_at'] = datetime.now(timezone.utc).isoformat()
                updated.append(document)
            # Failure during index/model building leaves the old corpus untouched.
            candidate = EnterpriseRAGPipeline(self.config, updated)
            content = ''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in updated)
            atomic_write(self.config.corpus_path, content)
            self._pipeline = candidate
            self._signature = self._stat()
            # Remove previous index snapshots, including deleted document text.
            for path in self.config.index_dir.glob('*.npz'):
                if path.stem != candidate.corpus_revision:
                    path.unlink(missing_ok=True)
            event = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'operation': 'delete' if delete_id else 'update' if exists else 'create',
                'doc_id': doc_id, 'previous_revision': previous,
                'corpus_revision': candidate.corpus_revision, 'pii_redacted': redacted,
                'documents': len(updated),
                'content_sha256': hashlib.sha256(document['text'].encode()).hexdigest()
                                  if document else None,
            }
            with self.events_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            return event

    def list_documents(self):
        pipeline = self.current()
        return {'corpus_revision': pipeline.corpus_revision,
                'documents': [{k: v for k, v in d.items() if k != 'text'}
                              | {'characters': len(d['text'])} for d in pipeline.documents]}
