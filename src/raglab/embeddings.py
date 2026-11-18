"""Turning text into vectors.

An embedding model maps a piece of text to a list of numbers (a vector) such
that texts with similar meaning end up close together. "Close" is usually
measured with cosine similarity.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Sequence

import numpy as np

DEFAULT_MODEL = os.getenv("RAGLAB_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# BGE models were trained with this instruction in front of *queries* (not passages).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between vector(s) ``a`` and the rows of ``b``.

    cos(a, b) = (a . b) / (|a| * |b|). 1 = same direction, 0 = unrelated.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a_norm = a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12, None)
    return a_norm @ b_norm.T


def _cache_dir() -> Path:
    path = Path(os.getenv("RAGLAB_CACHE", Path.home() / ".cache" / "raglab"))
    path.mkdir(parents=True, exist_ok=True)
    return path


class Embedder:
    """Thin wrapper around a sentence-transformers model with a disk cache."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, done lazily

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        self.dim = self.model.get_sentence_embedding_dimension()
        self.query_prefix = BGE_QUERY_PREFIX if "bge" in model_name.lower() else ""

    def embed_passages(self, texts: Sequence[str], use_cache: bool = True) -> np.ndarray:
        """Embed documents/chunks. Results are cached on disk by content hash."""
        key = hashlib.sha256((self.model_name + "\x00".join(texts)).encode()).hexdigest()[:24]
        path = _cache_dir() / f"emb-{key}.npy"
        if use_cache and path.exists():
            return np.load(path)
        vectors = self.model.encode(list(texts), normalize_embeddings=True,
                                    batch_size=32, show_progress_bar=len(texts) > 64)
        vectors = np.asarray(vectors, dtype=np.float32)
        if use_cache:
            np.save(path, vectors)
        return vectors

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single question (with the model's query instruction, if any)."""
        vector = self.model.encode([self.query_prefix + text], normalize_embeddings=True)[0]
        return np.asarray(vector, dtype=np.float32)


_SHARED: dict[str, Embedder] = {}


def get_embedder(model_name: str = DEFAULT_MODEL) -> Embedder:
    """Load a model once per process and reuse it."""
    if model_name not in _SHARED:
        _SHARED[model_name] = Embedder(model_name)
    return _SHARED[model_name]
