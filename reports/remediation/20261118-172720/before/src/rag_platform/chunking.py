"""Four enterprise chunking strategies for RAG ablation benchmarking."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Protocol, Sequence

import numpy as np

from .embeddings import SemanticEmbedder


@dataclass
class Chunk:
    """Represents an indexed retrieval chunk and its enclosing context."""

    chunk_id: str
    doc_id: str
    title: str
    domain: str
    text: str
    parent_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Chunker(Protocol):
    """Protocol implemented by all chunking strategies."""

    name: str

    def chunk_documents(self, documents: Sequence[dict[str, Any]]) -> list[Chunk]:
        ...


def _split_sentences(text: str) -> list[str]:
    """Split markdown/prose into clean sentences while preserving table lines."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    units: list[str] = []
    for ln in lines:
        if ln.startswith("|") or ln.startswith("#"):
            units.append(ln)
        else:
            parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`])", ln)
            for p in parts:
                if p.strip():
                    units.append(p.strip())
    return units


class FixedWindowChunker:
    """Naive fixed-character sliding window chunker (baseline)."""

    name = "fixed"

    def __init__(self, chunk_size: int = 400, overlap: int = 80) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_documents(self, documents: Sequence[dict[str, Any]]) -> list[Chunk]:
        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        for doc in documents:
            text = doc["text"]
            doc_id = doc["doc_id"]
            idx = 0
            start = 0
            while start < len(text):
                segment = text[start : start + self.chunk_size].strip()
                if segment:
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc_id}::fixed::{idx}",
                            doc_id=doc_id,
                            title=doc["title"],
                            domain=doc["domain"],
                            text=segment,
                            parent_text=segment,
                            metadata={"strategy": self.name, "chunk_index": idx},
                        )
                    )
                    idx += 1
                if start + self.chunk_size >= len(text):
                    break
                start += step
        return chunks


class RecursiveMarkdownChunker:
    """Structure-aware recursive splitter preserving markdown headings and paragraphs."""

    name = "recursive"

    def __init__(self, max_chars: int = 500) -> None:
        self.max_chars = max_chars

    def _split_recursive(self, text: str) -> list[str]:
        if len(text) <= self.max_chars:
            return [text.strip()] if text.strip() else []

        # Try splitting on markdown H2 headings, then double newlines, then sentences
        for sep in ("\n## ", "\n\n", "\n", ". "):
            if sep in text:
                parts = text.split(sep)
                grouped: list[str] = []
                buf = ""
                for i, part in enumerate(parts):
                    piece = (sep.lstrip("\n") + part) if (sep == "\n## " and i > 0) else part
                    piece = piece.strip()
                    if not piece:
                        continue
                    candidate = f"{buf}\n{piece}".strip() if buf else piece
                    if len(candidate) <= self.max_chars:
                        buf = candidate
                    else:
                        if buf:
                            grouped.append(buf)
                        if len(piece) > self.max_chars:
                            grouped.extend(self._hard_split(piece))
                            buf = ""
                        else:
                            buf = piece
                if buf:
                    grouped.append(buf)
                if grouped:
                    return grouped

        return self._hard_split(text)

    def _hard_split(self, text: str) -> list[str]:
        return [
            text[i : i + self.max_chars].strip()
            for i in range(0, len(text), self.max_chars)
            if text[i : i + self.max_chars].strip()
        ]

    def chunk_documents(self, documents: Sequence[dict[str, Any]]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            doc_id = doc["doc_id"]
            segments = self._split_recursive(doc["text"])
            for idx, seg in enumerate(segments):
                enriched = f"{doc['title']} | {seg}" if doc["title"] not in seg else seg
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}::rec::{idx}",
                        doc_id=doc_id,
                        title=doc["title"],
                        domain=doc["domain"],
                        text=enriched,
                        parent_text=enriched,
                        metadata={"strategy": self.name, "chunk_index": idx},
                    )
                )
        return chunks


class SemanticBreakpointChunker:
    """Splits text at sentence boundaries where adjacent cosine similarity drops sharply."""

    name = "semantic"

    def __init__(
        self,
        embedder: SemanticEmbedder | None = None,
        breakpoint_percentile: float = 80.0,
        max_chars: int = 650,
    ) -> None:
        self.embedder = embedder
        self.breakpoint_percentile = breakpoint_percentile
        self.max_chars = max_chars

    def chunk_documents(self, documents: Sequence[dict[str, Any]]) -> list[Chunk]:
        all_sentences: list[str] = []
        doc_sentences: list[list[str]] = []
        for doc in documents:
            sents = _split_sentences(doc["text"])
            doc_sentences.append(sents)
            all_sentences.extend(sents)

        embedder = self.embedder
        if embedder is None or not embedder.is_fitted:
            embedder = SemanticEmbedder(embedding_dim=64)
            embedder.fit(all_sentences if all_sentences else [d["text"] for d in documents])

        chunks: list[Chunk] = []
        for doc, sents in zip(documents, doc_sentences):
            doc_id = doc["doc_id"]
            if len(sents) <= 2:
                joined = "\n".join(sents)
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}::sem::0",
                        doc_id=doc_id,
                        title=doc["title"],
                        domain=doc["domain"],
                        text=joined,
                        parent_text=joined,
                        metadata={"strategy": self.name, "chunk_index": 0},
                    )
                )
                continue

            vecs = embedder.encode(sents)
            # Cosine distance between adjacent sentences (since vecs are L2-normalized)
            sims = np.sum(vecs[:-1] * vecs[1:], axis=1)
            distances = 1.0 - sims
            threshold = float(np.percentile(distances, self.breakpoint_percentile))

            groups: list[list[str]] = [[sents[0]]]
            for idx in range(1, len(sents)):
                cur_len = sum(len(s) for s in groups[-1])
                should_break = (distances[idx - 1] >= threshold and cur_len >= 160) or (
                    cur_len + len(sents[idx]) > self.max_chars
                )
                if should_break:
                    groups.append([sents[idx]])
                else:
                    groups[-1].append(sents[idx])

            for idx, grp in enumerate(groups):
                text = "\n".join(grp).strip()
                if doc["title"] not in text:
                    text = f"{doc['title']}\n{text}"
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}::sem::{idx}",
                        doc_id=doc_id,
                        title=doc["title"],
                        domain=doc["domain"],
                        text=text,
                        parent_text=text,
                        metadata={"strategy": self.name, "chunk_index": idx},
                    )
                )
        return chunks


class ParentChildChunker:
    """Hierarchical chunker: indexes granular child snippets, returns rich parent context."""

    name = "parent_child"

    def __init__(self, parent_max_chars: int = 900, child_max_chars: int = 260) -> None:
        self.parent_max_chars = parent_max_chars
        self.child_max_chars = child_max_chars
        self._parent_splitter = RecursiveMarkdownChunker(max_chars=parent_max_chars)
        self._child_splitter = RecursiveMarkdownChunker(max_chars=child_max_chars)

    def chunk_documents(self, documents: Sequence[dict[str, Any]]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            doc_id = doc["doc_id"]
            parents = self._parent_splitter._split_recursive(doc["text"])
            child_counter = 0
            for p_idx, parent_raw in enumerate(parents):
                parent_full = (
                    f"[{doc_id}] {doc['title']} (Domain: {doc['domain']})\n{parent_raw}"
                )
                children = self._child_splitter._split_recursive(parent_raw)
                for c_sub_idx, child_raw in enumerate(children):
                    child_text = f"{doc['title']} ({doc_id}) | {child_raw}"
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc_id}::p{p_idx}::c{child_counter}",
                            doc_id=doc_id,
                            title=doc["title"],
                            domain=doc["domain"],
                            text=child_text,
                            parent_text=parent_full,
                            metadata={
                                "strategy": self.name,
                                "parent_index": p_idx,
                                "child_index": c_sub_idx,
                            },
                        )
                    )
                    child_counter += 1
        return chunks


def build_chunker(strategy: str, **kwargs: Any) -> Chunker:
    """Factory returning a chunker by strategy name."""
    s = strategy.lower().strip()
    if s == "fixed":
        return FixedWindowChunker(
            chunk_size=int(kwargs.get("fixed_chunk_size", 400)),
            overlap=int(kwargs.get("fixed_overlap", 80)),
        )
    if s == "recursive":
        return RecursiveMarkdownChunker(max_chars=int(kwargs.get("recursive_max_chars", 500)))
    if s == "semantic":
        return SemanticBreakpointChunker(
            embedder=kwargs.get("embedder"),
            breakpoint_percentile=float(kwargs.get("semantic_breakpoint_percentile", 80.0)),
        )
    if s == "parent_child":
        return ParentChildChunker(
            parent_max_chars=int(kwargs.get("parent_max_chars", 900)),
            child_max_chars=int(kwargs.get("child_max_chars", 260)),
        )
    raise ValueError(f"Unknown chunking strategy: {strategy}")
