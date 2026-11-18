"""Splitting documents into chunks.

Why chunk at all? Embedding models and LLM prompts have size limits, and a
small, focused piece of text matches a question better than a whole manual.
The trade-off: chunks that are too small lose context; chunks that are too big
dilute the signal.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .data import Document


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    #: Larger surrounding text returned to the LLM (parent-child chunking).
    context: str = ""

    def __post_init__(self) -> None:
        if not self.context:
            self.context = self.text


# --- Strategy 1: fixed-size windows -------------------------------------

def fixed_size_chunks(text: str, size: int = 500, overlap: int = 100) -> list[str]:
    """Cut ``text`` into windows of ``size`` characters.

    Consecutive windows share ``overlap`` characters so that a sentence cut at
    a boundary still appears whole in one of the two neighbours.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    step = size - overlap
    pieces = []
    for start in range(0, len(text), step):
        piece = text[start:start + size]
        if piece.strip():
            pieces.append(piece)
        if start + size >= len(text):
            break
    return pieces


# --- Strategy 2: recursive splitting on natural boundaries ---------------

DEFAULT_SEPARATORS = ("\n## ", "\n\n", "\n", ". ", " ")


def recursive_chunks(text: str, max_chars: int = 800,
                     separators: tuple[str, ...] = DEFAULT_SEPARATORS) -> list[str]:
    """Split on the "biggest" boundary first (headings), then paragraphs,
    lines, sentences and finally words, until every piece fits ``max_chars``.
    Small neighbouring pieces are merged back together.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    for i, sep in enumerate(separators):
        if sep not in text:
            continue
        parts = text.split(sep)
        # Keep the separator with the part that follows it (e.g. the "## " heading).
        parts = [parts[0]] + [sep.lstrip("\n") + p if sep.startswith("\n") else p for p in parts[1:]]
        merged: list[str] = []
        current = ""
        # Re-join with exactly the text that was removed, so chunks stay verbatim substrings.
        joiner = sep[:len(sep) - len(sep.lstrip("\n"))] if sep.startswith("\n") else sep
        for part in parts:
            candidate = f"{current}{joiner}{part}" if current else part
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    merged.append(current)
                current = part
        if current:
            merged.append(current)
        result: list[str] = []
        for piece in merged:
            if len(piece) > max_chars:
                result.extend(recursive_chunks(piece, max_chars, separators[i + 1:] or (" ",)))
            elif piece.strip():
                result.append(piece.strip())
        return result
    # No separator at all: fall back to hard cuts.
    return fixed_size_chunks(text, max_chars, 0)


# --- Strategy 3: one chunk per markdown section --------------------------

def section_chunks(text: str) -> list[str]:
    """One chunk per ``## `` section; the text before the first one is the intro."""
    sections = re.split(r"\n(?=## )", text)
    return [s.strip() for s in sections if s.strip()]


# --- Putting it together --------------------------------------------------

def chunk_documents(docs: Iterable[Document], method: str = "recursive",
                    **kwargs) -> list[Chunk]:
    """Chunk a corpus.

    ``method`` is one of ``"fixed"``, ``"recursive"``, ``"section"`` or
    ``"parent_child"`` (small recursive children that carry their whole
    section as ``context``).
    """
    chunks: list[Chunk] = []
    for doc in docs:
        if method == "fixed":
            pairs = [(p, p) for p in fixed_size_chunks(doc.text, kwargs.get("size", 500), kwargs.get("overlap", 100))]
        elif method == "recursive":
            pairs = [(p, p) for p in recursive_chunks(doc.text, kwargs.get("max_chars", 800))]
        elif method == "section":
            pairs = [(p, p) for p in section_chunks(doc.text)]
        elif method == "parent_child":
            pairs = []
            for parent in section_chunks(doc.text):
                for child in recursive_chunks(parent, kwargs.get("max_chars", 300)):
                    pairs.append((child, parent))
        else:
            raise ValueError(f"unknown chunking method: {method}")
        for i, (text, context) in enumerate(pairs):
            chunks.append(Chunk(f"{doc.doc_id}#{i}", doc.doc_id, doc.title, text, context))
    return chunks
