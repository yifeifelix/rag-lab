"""Fixed-encoder vector reuse and incremental Qdrant REST indexing.

Qdrant holds immutable vector IDs and minimal metadata, never source text. Each
snapshot filters to its own IDs; retired snapshots use their in-memory vectors.
"""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading
import uuid

import httpx
import numpy as np

from .retrieval import DenseVectorIndex, tokenize_technical


def chunk_key(chunk):
    return hashlib.sha256(json.dumps(asdict(chunk), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def encode_cached(chunks, embedder, config):
    """Only encode unseen texts in a fixed, versioned embedding space."""
    identity = f'{config.embedding_model}@{embedder.revision}'
    namespace = hashlib.sha256(identity.encode()).hexdigest()
    path = config.index_dir / 'vectors.sqlite3'
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE IF NOT EXISTS vectors (space TEXT, hash TEXT, vector BLOB, PRIMARY KEY(space,hash))')
        keys = [hashlib.sha256(c.text.encode()).hexdigest() for c in chunks]
        stored = dict(connection.execute('SELECT hash,vector FROM vectors WHERE space=?', (namespace,)))
        missing = list(dict.fromkeys(key for key in keys if key not in stored))
        text_by_key = dict(zip(keys, (c.text for c in chunks)))
        if missing:
            values = embedder.encode([text_by_key[key] for key in missing])
            for key, vector in zip(missing, values):
                stored[key] = np.asarray(vector, dtype=np.float32).tobytes()
                connection.execute('INSERT OR REPLACE INTO vectors VALUES (?,?,?)', (namespace, key, stored[key]))
        # Remove orphan vectors on the next successful build; they contain no source text.
        active = set(keys)
        connection.executemany('DELETE FROM vectors WHERE space=? AND hash=?',
                               [(namespace, key) for key in stored if key not in active])
    vectors = np.stack([np.frombuffer(stored[key], dtype=np.float32) for key in keys]) if keys else np.zeros((0, embedder.embedding_dim), dtype=np.float32)
    if vectors.shape != (len(chunks), embedder.embedding_dim) or not np.isfinite(vectors).all():
        raise ValueError('Incompatible vector cache; rebuild the disposable index')
    return vectors, len(missing)


class IncrementalBM25Index:
    """Copy-on-write postings; only changed chunks are tokenized and indexed."""

    def __init__(self, chunks, previous=None, k1=1.5, b=0.75):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.keys = [chunk_key(chunk) for chunk in chunks]
        self.positions = {key: index for index, key in enumerate(self.keys)}
        self.records = dict(previous.records) if previous else {}
        self.postings = dict(previous.postings) if previous else {}
        active = set(self.keys)
        added = active - self.records.keys()
        removed = self.records.keys() - active
        new_records = {key: Counter(tokenize_technical(chunks[self.positions[key]].text)) for key in added}
        touched = set().union(*(self.records[key].keys() for key in removed),
                              *(record.keys() for record in new_records.values()))
        for term in touched:
            posting = dict(self.postings.get(term, {}))
            for key in removed:
                posting.pop(key, None)
            for key, counts in new_records.items():
                if term in counts:
                    posting[key] = counts[term]
            if posting:
                self.postings[term] = posting
            else:
                self.postings.pop(term, None)
        for key in removed:
            self.records.pop(key)
        self.records.update(new_records)
        self.lengths = {key: sum(counts.values()) for key, counts in self.records.items()}
        self.avgdl = sum(self.lengths.values()) / max(1, len(chunks)) or 1.0
        self.vocab = {term: index for index, term in enumerate(sorted(self.postings))}
        self.tokenized_chunks = len(added)

    def search(self, query, top_k=20, domain_filter=None):
        scores = {}
        n = len(self.chunks)
        for term in set(tokenize_technical(query)):
            postings = self.postings.get(term, {})
            idf = math.log(1 + (n - len(postings) + 0.5) / (len(postings) + 0.5))
            for key, count in postings.items():
                index = self.positions[key]
                if domain_filter and self.chunks[index].domain != domain_filter:
                    continue
                norm = 1 - self.b + self.b * self.lengths[key] / self.avgdl
                scores[index] = scores.get(index, 0.0) + idf * count * (self.k1 + 1) / (count + self.k1 * norm)
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]


class QdrantDenseIndex(DenseVectorIndex):
    def __init__(self, chunks, embedder, embeddings, config):
        super().__init__(chunks, embedder, embeddings)
        if not embedder.revision:
            raise ValueError('Qdrant requires an immutable embedding revision or local model digest')
        namespace = json.dumps([str(config.corpus_path.resolve()), config.embedding_model,
                                embedder.revision, embedder.embedding_dim])
        suffix = hashlib.sha256(namespace.encode()).hexdigest()[:16]
        self.collection = f'{config.qdrant_collection}_{suffix}'
        self.client = httpx.Client(base_url=config.qdrant_url, timeout=15, trust_env=False,
                                   headers={'api-key': __import__('os').getenv('QDRANT_API_KEY', '')})
        self.ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_key(chunk))) for chunk in chunks]
        self.id_to_index = {value: index for index, value in enumerate(self.ids)}
        self._lock = threading.RLock()
        self._remote = True
        response = self.client.get(f'/collections/{self.collection}')
        if response.status_code == 404:
            self._request('PUT', '', {'vectors': {'size': embedder.embedding_dim, 'distance': 'Cosine'}})
        else:
            response.raise_for_status()
            vectors = response.json()['result']['config']['params']['vectors']
            if vectors['size'] != embedder.embedding_dim:
                raise ValueError('Qdrant vector dimensions do not match the fixed encoder')
        existing = self._all_ids()
        missing = set(self.ids) - existing
        points = [{'id': value, 'vector': embeddings[index].tolist(),
                   'payload': {'doc_id': chunks[index].doc_id, 'domain': chunks[index].domain}}
                  for index, value in enumerate(self.ids) if value in missing]
        for start in range(0, len(points), 128):
            self._request('PUT', '/points?wait=true', {'points': points[start:start + 128]})
        self._obsolete = existing - set(self.ids)
        self.upserted_points = len(points)

    def _request(self, method, suffix, payload):
        response = self.client.request(method, f'/collections/{self.collection}{suffix}', json=payload)
        response.raise_for_status()
        return response.json().get('result')

    def _all_ids(self):
        ids, offset = set(), None
        while True:
            result = self._request('POST', '/points/scroll', {'limit': 256, 'offset': offset,
                                                             'with_payload': False, 'with_vector': False})
            ids.update(str(point['id']) for point in result['points'])
            offset = result.get('next_page_offset')
            if offset is None:
                return ids

    def search_vector(self, query_vec, top_k=20, domain_filter=None):
        with self._lock:
            if not self._remote:
                return super().search_vector(query_vec, top_k, domain_filter)
            if not self.ids or not np.any(query_vec):
                return []
            must = [{'has_id': self.ids}]
            if domain_filter:
                must.append({'key': 'domain', 'match': {'value': domain_filter}})
            result = self._request('POST', '/points/query', {'query': query_vec.tolist(),
                    'filter': {'must': must}, 'limit': top_k, 'with_payload': False})
            return [(self.id_to_index[str(point['id'])], float(point['score']))
                    for point in result['points'] if str(point['id']) in self.id_to_index]

    def retire(self):
        # Wait for a remote read already in progress before physical deletion.
        with self._lock:
            self._remote = False
            self.client.close()

    def ready(self):
        try:
            return self.client.get(f'/collections/{self.collection}', timeout=2).is_success
        except (httpx.HTTPError, RuntimeError):
            return False

    def activate(self):
        if self._obsolete:
            self._request('POST', '/points/delete?wait=true', {'points': sorted(self._obsolete)})
            self._obsolete.clear()

    def close(self):
        self.client.close()
