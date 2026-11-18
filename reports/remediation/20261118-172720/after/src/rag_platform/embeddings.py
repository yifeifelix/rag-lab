"""Deterministic semantic embedding engine with optional cloud provider adapters."""

from __future__ import annotations

import math
import os
import re
from typing import Sequence

import httpx
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

_TOKEN_RE = re.compile(r"(?u)\b[a-zA-Z]{2,}[a-zA-Z0-9]*\b")


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize a 2D array row-wise with zero-vector safety."""
    if vectors.size == 0:
        return vectors.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    return (vectors / safe_norms).astype(np.float32)


class SemanticEmbedder:
    """Offline lexical baseline: word TF-IDF (1, 2)-grams + TruncatedSVD."""

    def __init__(self, embedding_dim: int = 48, random_state: int = 42) -> None:
        self.embedding_dim = embedding_dim
        self.random_state = random_state
        self.word_vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        self.is_fitted: bool = False
        self.explained_variance_ratio_sum: float = 0.0

    def fit(self, texts: Sequence[str]) -> "SemanticEmbedder":
        """Fit word-level TF-IDF vectorizer and TruncatedSVD latent projection."""
        cleaned = [t.strip() if t.strip() else "empty" for t in texts]
        min_df = 2 if len(cleaned) >= 20 else 1
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            lowercase=True,
            min_df=min_df,
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z]{2,}[a-zA-Z0-9]*\b",
            max_features=8000,
        )
        w_mat = self.word_vectorizer.fit_transform(cleaned)

        max_components = min(self.embedding_dim, w_mat.shape[0] - 1, w_mat.shape[1] - 1)
        n_comp = max(1, max_components)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        self.svd.fit(w_mat)
        self.explained_variance_ratio_sum = float(np.sum(self.svd.explained_variance_ratio_))
        # Precompute fast single-query inference arrays.
        self._vocab: dict[str, int] = dict(self.word_vectorizer.vocabulary_)
        self._idf: np.ndarray = self.word_vectorizer.idf_.astype(np.float32)
        self._comp_T: np.ndarray = self.svd.components_.T.astype(np.float32)
        self.is_fitted = True
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a sequence of strings into L2-normalized float32 embeddings."""
        if not self.is_fitted or self.svd is None or self.word_vectorizer is None:
            raise RuntimeError("SemanticEmbedder must be fitted before calling encode().")
        if not texts:
            return np.zeros((0, self.svd.n_components), dtype=np.float32)

        cleaned = [t.strip() if t.strip() else "empty" for t in texts]
        w_mat = self.word_vectorizer.transform(cleaned)
        dense = self.svd.transform(w_mat)
        return _l2_normalize(dense)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single query string into a 1D normalized embedding vector (~8 microseconds)."""
        if not self.is_fitted or self.svd is None:
            raise RuntimeError("SemanticEmbedder must be fitted before calling encode_single().")
        words = [w for w in _TOKEN_RE.findall(text.lower()) if w not in ENGLISH_STOP_WORDS]
        counts: dict[int, int] = {}
        vocab = self._vocab
        for i, w in enumerate(words):
            idx = vocab.get(w)
            if idx is not None:
                counts[idx] = counts.get(idx, 0) + 1
            if i + 1 < len(words):
                bidx = vocab.get(f"{w} {words[i + 1]}")
                if bidx is not None:
                    counts[bidx] = counts.get(bidx, 0) + 1
        if not counts:
            return np.zeros(self._comp_T.shape[1], dtype=np.float32)
        idxs = np.fromiter(counts.keys(), dtype=np.int32)
        tfs = np.fromiter(counts.values(), dtype=np.float32)
        weights = (1.0 + np.log(tfs)) * self._idf[idxs]
        wnorm = float(np.linalg.norm(weights))
        if wnorm > 1e-12:
            weights /= wnorm
        proj = weights @ self._comp_T[idxs]
        pnorm = float(np.linalg.norm(proj))
        if pnorm > 1e-12:
            proj /= pnorm
        return proj.astype(np.float32)


class PretrainedEmbedder:
    """Optional pretrained sentence encoder; downloads require explicit opt-in."""

    svd = None
    is_fitted = True

    def __init__(self, model: str, allow_download: bool = False, revision: str = ''):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError('Install the pretrained optional dependency first: pip install -e ".[pretrained]"') from exc
        self.model = SentenceTransformer(model, local_files_only=not allow_download,
                                         revision=revision or None, trust_remote_code=False,
                                         cache_folder=os.getenv('HF_HOME'), device='cpu')
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        first = self.model._first_module()
        self.revision = revision or getattr(getattr(getattr(first, 'auto_model', None), 'config', None), '_commit_hash', '')
        if not self.revision and __import__('pathlib').Path(model).is_dir():
            import hashlib
            from pathlib import Path
            digest = hashlib.sha256()
            for path in sorted(Path(model).rglob('*')):
                if path.is_file():
                    digest.update(str(path.relative_to(model)).encode())
                    with path.open('rb') as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b''):
                            digest.update(block)
            self.revision = digest.hexdigest()

    def fit(self, texts):
        return self

    def encode(self, texts):
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True,
                                           show_progress_bar=False), dtype=np.float32)

    def encode_single(self, text):
        return self.encode([text])[0]


def build_embedder(config):
    if config.embedding_backend == 'sentence_transformers':
        return PretrainedEmbedder(config.embedding_model, config.allow_model_download, config.embedding_revision)
    return SemanticEmbedder(embedding_dim=config.embedding_dim)


class CloudEmbeddingAdapter:
    """Optional remote embedding adapter for OpenAI or Gemini when API keys are supplied."""

    def __init__(self, provider: str = "openai", model: str | None = None) -> None:
        self._client = None
        self.provider = provider.lower()
        self.model = model or (
            "text-embedding-3-small" if self.provider == "openai" else "text-embedding-004"
        )

    async def encode_async(self, texts: Sequence[str]) -> np.ndarray:
        """Call remote embedding API asynchronously."""
        from .security import redact_pii
        texts = [redact_pii(text)[0] for text in texts]
        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set.")
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(timeout=15.0, trust_env=False)
            if self._client is not None:
                client = self._client
                resp = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"input": list(texts), "model": self.model},
                )
                resp.raise_for_status()
                data = resp.json()["data"]
                vecs = np.array([item["embedding"] for item in data], dtype=np.float32)
                return _l2_normalize(vecs)

        if self.provider == "gemini":
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', self.model):
                raise ValueError('Invalid Gemini embedding model name')
            api_key = os.getenv("GEMINI_API_KEY", "")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is not set.")
            if self._client is None or self._client.is_closed:
                self._client = httpx.AsyncClient(timeout=15.0, trust_env=False)
            if self._client is not None:
                client = self._client
                requests_payload = [
                    {"model": f"models/{self.model}", "content": {"parts": [{"text": t}]}}
                    for t in texts
                ]
                resp = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:batchEmbedContents",
                    headers={"x-goog-api-key": api_key},
                    json={"requests": requests_payload},
                )
                resp.raise_for_status()
                vecs = np.array(
                    [item["values"] for item in resp.json()["embeddings"]],
                    dtype=np.float32,
                )
                return _l2_normalize(vecs)

        raise ValueError(f"Unsupported embedding provider: {self.provider}")

    async def aclose(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
