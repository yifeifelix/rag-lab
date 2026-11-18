"""Small helpers to show results nicely inside notebooks."""

from __future__ import annotations

import html

import pandas as pd


def hits_table(hits, max_chars: int = 160) -> pd.DataFrame:
    """Retrieval results as a table: rank, score, document, text preview."""
    return pd.DataFrame([{
        "rank": h.rank, "score": round(h.score, 4), "doc_id": h.doc_id,
        "preview": " ".join(h.chunk.text.split())[:max_chars],
    } for h in hits]).set_index("rank")


def show_chunks(chunks, n: int = 5, max_chars: int = 400):
    """Render the first ``n`` chunks as boxes so boundaries are easy to see."""
    from IPython.display import HTML

    boxes = []
    for c in chunks[:n]:
        text = html.escape(c.text[:max_chars]) + ("…" if len(c.text) > max_chars else "")
        boxes.append(
            f"<div style='border:1px solid #8888;border-radius:6px;padding:8px;margin:6px 0;"
            f"white-space:pre-wrap;font-family:monospace;font-size:12px'>"
            f"<b>{html.escape(c.chunk_id)}</b> · {len(c.text)} chars<br>{text}</div>")
    return HTML("".join(boxes))


def highlight(text: str, words) -> object:
    """Show ``text`` with the given words highlighted."""
    from IPython.display import HTML
    import re

    escaped = html.escape(text)
    for word in sorted(set(words), key=len, reverse=True):
        escaped = re.sub(f"(?i)({re.escape(html.escape(word))})", r"<mark>\1</mark>", escaped)
    return HTML(f"<div style='white-space:pre-wrap'>{escaped}</div>")
