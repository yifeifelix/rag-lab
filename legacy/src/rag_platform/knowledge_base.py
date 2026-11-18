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
import weakref

from .data_gen import generate_datasets, load_corpus_documents
from .pipeline import EnterpriseRAGPipeline
from .observability import event
import logging


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
        self._write_lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._retired = weakref.WeakSet()
        self.lock_path = config.corpus_path.with_suffix('.lock')
        self.events_path = config.corpus_path.parent / 'document_events.jsonl'
        config.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        with self._file_lock():
            from .bootstrap import initialize_workspace
            initialize_workspace(config)
            self._pipeline = EnterpriseRAGPipeline(config, load_corpus_documents(config.corpus_path))
            self.config = self._pipeline.config
            self._signature = self._stat()
            self._activate(self._pipeline)

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

    @staticmethod
    def _activate(pipeline):
        dense = pipeline.retriever.dense_index
        if hasattr(dense, 'activate'):
            try:
                dense.activate()
            except Exception as exc:
                # Canonical JSON is already committed. Filtered reads remain safe;
                # orphan cleanup is retried by the next startup/update.
                event('vector_cleanup_failed', level=logging.WARNING, error_type=type(exc).__name__)

    def _publish(self, candidate):
        previous = self._pipeline
        dense = previous.retriever.dense_index
        if hasattr(dense, 'retire'):
            dense.retire()
        with self._lock:
            self._retired.add(previous)
            self._pipeline = candidate
            self.config = candidate.config
            self._signature = self._stat()
        self._activate(candidate)

    def current(self):
        with self._lock:
            snapshot, signature = self._pipeline, self._signature
        if self._stat() == signature:
            return snapshot
        # One caller refreshes outside the reader lock; concurrent readers keep
        # using the immutable previous snapshot instead of waiting for a rebuild.
        if not self._refresh_lock.acquire(blocking=False):
            return snapshot
        try:
            with self._write_lock, self._file_lock():
                if self._stat() != self._signature:
                    candidate = EnterpriseRAGPipeline(self.config,
                        load_corpus_documents(self.config.corpus_path), previous=self._pipeline)
                    self._publish(candidate)
            return self._pipeline
        finally:
            self._refresh_lock.release()

    def mutate(self, document=None, delete_id=None, expected_revision=None):
        # Serialize writers, but do not hold the reader lock during chunking,
        # embedding, filesystem I/O or remote vector operations.
        with self._write_lock, self._file_lock():
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
            if document is not None:
                document = dict(document)
                # Preserve exact source text. Privacy filtering belongs to query,
                # provider egress and logging, not canonical knowledge storage.
                document['updated_at'] = datetime.now(timezone.utc).isoformat()
                updated.append(document)
            candidate = EnterpriseRAGPipeline(self.config, updated, previous=self._pipeline)
            content = ''.join(json.dumps(d, ensure_ascii=False) + '\n' for d in updated)
            atomic_write(self.config.corpus_path, content)
            self._publish(candidate)
            for path in self.config.index_dir.glob('*.npz'):
                if path.stem != candidate.corpus_revision:
                    path.unlink(missing_ok=True)
            record = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'operation': 'delete' if delete_id else 'update' if exists else 'create',
                'doc_id': doc_id, 'previous_revision': previous,
                'corpus_revision': candidate.corpus_revision, 'pii_redacted': False,
                'documents': len(updated),
                'content_sha256': hashlib.sha256(document['text'].encode()).hexdigest() if document else None,
            }
            with self.events_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            event('document_mutation', operation=record['operation'], documents=len(updated),
                  corpus_revision=candidate.corpus_revision)
            return record

    async def aclose(self):
        for pipeline in [self._pipeline, *self._retired]:
            await pipeline.aclose()
        self._retired.clear()

    def list_documents(self):
        pipeline = self.current()
        return {'corpus_revision': pipeline.corpus_revision,
                'documents': [{k: v for k, v in d.items() if k != 'text'}
                              | {'characters': len(d['text'])} for d in pipeline.documents]}
