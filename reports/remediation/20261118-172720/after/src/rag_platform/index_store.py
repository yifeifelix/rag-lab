"""Content-addressed local index snapshots. JSON/NumPy only; never unpickle data."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .chunking import Chunk
from .embeddings import SemanticEmbedder


def fingerprint(documents, config) -> str:
    settings = {k: getattr(config, k) for k in (
        'chunking_strategy', 'fixed_chunk_size', 'fixed_overlap', 'recursive_max_chars',
        'semantic_breakpoint_percentile', 'parent_max_chars', 'child_max_chars',
        'embedding_dim', 'embedding_backend', 'embedding_model', 'embedding_revision', 'enable_contextual_headers')}
    from importlib.metadata import version
    code = {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ('chunking.py', 'embeddings.py', 'retrieval.py', 'index_store.py')}
    body = json.dumps({'schema': 1, 'documents': documents, 'settings': settings,
                       'code': code, 'sklearn': version('scikit-learn'), 'numpy': version('numpy')},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(body.encode()).hexdigest()


def load_index(path: Path, embedder):
    with np.load(path, allow_pickle=False) as saved:
        chunks = [Chunk(**c) for c in json.loads(str(saved['chunks']))]
        vectors = saved['vectors'].copy()
        if isinstance(embedder, SemanticEmbedder):
            vocabulary = json.loads(str(saved['vocabulary']))
            embedder.word_vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), sublinear_tf=True, lowercase=True,
                stop_words='english', token_pattern=r'(?u)\b[a-zA-Z]{2,}[a-zA-Z0-9]*\b',
                vocabulary=vocabulary)
            embedder.word_vectorizer.idf_ = saved['idf'].copy()
            embedder._vocab = vocabulary
            embedder._idf = saved['idf'].astype(np.float32)
            embedder.svd = TruncatedSVD(n_components=saved['components'].shape[0])
            embedder.svd.components_ = saved['components'].copy()
            embedder.svd.n_features_in_ = saved['components'].shape[1]
            embedder._comp_T = embedder.svd.components_.T.astype(np.float32)
            embedder.explained_variance_ratio_sum = float(saved['explained'])
            embedder.is_fitted = True
        return chunks, vectors


def save_index(path: Path, chunks, embedder, vectors) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {'chunks': np.array(json.dumps([asdict(c) for c in chunks], ensure_ascii=False)),
             'vectors': vectors}
    if isinstance(embedder, SemanticEmbedder):
        state.update(vocabulary=np.array(json.dumps({k: int(v) for k, v in embedder._vocab.items()})),
                     idf=embedder.word_vectorizer.idf_, components=embedder.svd.components_,
                     explained=np.array(embedder.explained_variance_ratio_sum))
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.npz')
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, **state)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
