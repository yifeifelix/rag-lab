"""Multi-tier caching (L1 Exact, L2 Semantic Vector) and Singleflight request coalescing."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, replace
import hashlib
import time
from typing import Any, Awaitable, Callable, TypeVar

import numpy as np

T = TypeVar("T")


def normalize_query(query: str) -> str:
    """Canonicalize whitespace and casing for exact cache hashing."""
    return " ".join(query.strip().lower().split())


def make_cache_key(
    query: str,
    retrieval_mode: str,
    use_reranker: bool,
    top_k: int,
    domain_filter: str | None,
) -> str:
    """Build deterministic SHA-256 cache key scoped by retrieval parameters."""
    norm = normalize_query(query)
    raw = f"{norm}||{retrieval_mode}||{int(use_reranker)}||{top_k}||{domain_filter or '*'}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class L1ExactCache:
    """Bounded LRU + TTL exact-match cache."""

    def __init__(self, max_size: int = 4096, ttl_seconds: float = 3600.0) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self.hits: int = 0
        self.misses: int = 0

    def get(self, key: str) -> Any | None:
        item = self._store.get(key)
        if item is None:
            self.misses += 1
            return None
        expires_at, payload = item
        if time.monotonic() > expires_at:
            self._store.pop(key, None)
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return payload

    def set(self, key: str, payload: Any) -> None:
        expires_at = time.monotonic() + self.ttl_seconds
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (expires_at, payload)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._store)


@dataclass
class _SemanticEntry:
    scope: str
    vector: np.ndarray
    expires_at: float
    payload: Any


class L2SemanticCache:
    """Vector-similarity cache serving paraphrased queries without backend RAG/LLM calls."""

    def __init__(
        self,
        max_size: int = 2048,
        similarity_threshold: float = 0.90,
        ttl_seconds: float = 3600.0,
    ) -> None:
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self._entries: list[_SemanticEntry] = []
        # Keep stable entry references: list positions change on FIFO eviction.
        self._matrix_by_scope: dict[str, tuple[np.ndarray, list[_SemanticEntry]]] = {}
        self.hits: int = 0
        self.misses: int = 0

    def _rebuild_scope_matrix(self, scope: str) -> None:
        now = time.monotonic()
        entries = [e for e in self._entries if e.scope == scope and e.expires_at > now]
        if not entries:
            self._matrix_by_scope.pop(scope, None)
            return
        mat = np.vstack([e.vector for e in entries]).astype(np.float32)
        self._matrix_by_scope[scope] = (mat, entries)

    def get(
        self,
        query_vec: np.ndarray,
        scope: str,
    ) -> tuple[Any, float] | None:
        scope_data = self._matrix_by_scope.get(scope)
        if scope_data is None:
            self.misses += 1
            return None

        mat, entries = scope_data
        if any(e.expires_at <= time.monotonic() for e in entries):
            self._rebuild_scope_matrix(scope)
            return self.get(query_vec, scope)
        sims = mat @ query_vec
        best_pos = int(np.argmax(sims))
        best_sim = float(sims[best_pos])
        if best_sim < self.similarity_threshold:
            self.misses += 1
            return None

        entry = entries[best_pos]

        self.hits += 1
        return entry.payload, best_sim

    def set(self, query_vec: np.ndarray, scope: str, payload: Any) -> None:
        now = time.monotonic()
        if self.max_size <= 0:
            return
        # Evict oldest entry if full
        if len(self._entries) >= self.max_size:
            evicted = self._entries.pop(0)
            # Rebuild affected scopes cleanly
            scopes_to_refresh = {evicted.scope, scope}
            self._entries.append(
                _SemanticEntry(
                    scope=scope,
                    vector=query_vec.astype(np.float32),
                    expires_at=now + self.ttl_seconds,
                    payload=payload,
                )
            )
            for sc in scopes_to_refresh:
                self._rebuild_scope_matrix(sc)
            return

        self._entries.append(
            _SemanticEntry(
                scope=scope,
                vector=query_vec.astype(np.float32),
                expires_at=now + self.ttl_seconds,
                payload=payload,
            )
        )
        self._rebuild_scope_matrix(scope)

    def clear(self) -> None:
        self._entries.clear()
        self._matrix_by_scope.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)


class SingleflightCoalescer:
    """Coalesces concurrent identical in-flight requests to prevent thundering herds."""

    def __init__(self) -> None:
        self._inflight: dict[str, asyncio.Future[Any]] = {}
        self.coalesced_hits: int = 0
        self.leader_executions: int = 0

    async def execute(self, key: str, fn: Callable[[], Awaitable[T]]) -> tuple[T, bool]:
        """Run `fn` once per concurrent key; followers await leader's Future.

        Returns `(result, was_coalesced)`.
        """
        existing = self._inflight.get(key)
        if existing is not None:
            self.coalesced_hits += 1
            res = await asyncio.shield(existing)
            return res, True

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[T] = loop.create_future()
        self._inflight[key] = fut
        self.leader_executions += 1
        try:
            result = await fn()
            if not fut.done():
                fut.set_result(result)
            return result, False
        except BaseException as exc:
            if not fut.done():
                fut.set_exception(exc)
                fut.exception()
            raise
        finally:
            self._inflight.pop(key, None)

    def reset_stats(self) -> None:
        self.coalesced_hits = 0
        self.leader_executions = 0


def clone_dataclass_with(obj: Any, **updates: Any) -> Any:
    """Shallow-clone a dataclass payload with updated per-request telemetry fields."""
    return replace(obj, **updates)
