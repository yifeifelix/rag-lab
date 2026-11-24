"""Generate the lesson illustrations in course/images/ as plain SVG.

One picture per lesson, all in the same style. The three helpers keep their
colours everywhere: library = blue, librarian = green, writer = orange.

    python3 scripts/make_figures.py
"""

from __future__ import annotations

import math
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parents[1] / "course" / "images"

INK, MUTED, LINE = "#1f2937", "#6b7280", "#d1d5db"
BLUE = ("#dbeafe", "#2563eb")     # library
GREEN = ("#dcfce7", "#16a34a")    # librarian
ORANGE = ("#ffedd5", "#ea580c")   # writer
PURPLE = ("#ede9fe", "#7c3aed")
RED = ("#fee2e2", "#dc2626")
GREY = ("#f3f4f6", "#9ca3af")
YELLOW = ("#fef9c3", "#ca8a04")
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'SF Mono', Menlo, Consolas, monospace"


# --- tiny drawing helpers ---------------------------------------------------

def text(x, y, s, size=15, anchor="middle", weight="normal", fill=INK, italic=False, mono=False, halo=False):
    lines = s if isinstance(s, list) else [s]
    style = ' font-style="italic"' if italic else ""
    if halo:  # white outline keeps labels readable on top of lines
        style += ' stroke="#ffffff" stroke-width="4" paint-order="stroke" stroke-linejoin="round"'
    family = MONO if mono else FONT
    spans = "".join(
        f'<tspan x="{x}" dy="{0 if i == 0 else size * 1.3}">{escape(line)}</tspan>'
        for i, line in enumerate(lines))
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}" fill="{fill}"{style}>{spans}</text>')


def rect(x, y, w, h, colors=GREY, rx=10, dash=False, width=2):
    fill, stroke = colors
    d = ' stroke-dasharray="6 5"' if dash else ""
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{d}/>'


def arrow(x1, y1, x2, y2, color=MUTED, label=None, dash=False, both=False, label_dy=-8):
    marker = {MUTED: "am", INK: "ai", GREEN[1]: "ag", RED[1]: "ar", ORANGE[1]: "ao", BLUE[1]: "ab"}[color]
    d = ' stroke-dasharray="6 5"' if dash else ""
    start = f' marker-start="url(#{marker}s)"' if both else ""
    out = (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2.5"{d}'
           f' marker-end="url(#{marker})"{start}/>')
    if label:
        out += text((x1 + x2) / 2, (y1 + y2) / 2 + label_dy, label, 13, fill=color)
    return out


def path(d, color=MUTED, dash=False, end=True, width=2.5):
    marker = {MUTED: "am", INK: "ai", GREEN[1]: "ag", RED[1]: "ar", ORANGE[1]: "ao", BLUE[1]: "ab"}[color]
    dd = ' stroke-dasharray="6 5"' if dash else ""
    m = f' marker-end="url(#{marker})"' if end else ""
    return f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{dd}{m}/>'


def circle(x, y, r, colors=GREY, width=2, dash=False):
    fill, stroke = colors
    d = ' stroke-dasharray="6 5"' if dash else ""
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{d}/>'


def panel(x, y, w, h, colors, title, lines, title_size=17):
    out = rect(x, y, w, h, colors, rx=14)
    out += text(x + w / 2, y + 32, title, title_size, weight="bold", fill=colors[1])
    out += text(x + w / 2, y + 62, lines, 14)
    return out


def svg(name, w, h, body, title):
    markers = "".join(
        f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>'
        f'<marker id="{mid}s" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>'
        for mid, col in [("am", MUTED), ("ai", INK), ("ag", GREEN[1]), ("ar", RED[1]), ("ao", ORANGE[1]), ("ab", BLUE[1])])
    doc = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
           f'role="img" aria-label="{escape(title)}" style="max-width:100%;height:auto">'
           f'<title>{escape(title)}</title><defs>{markers}</defs>'
           f'<rect x="1" y="1" width="{w - 2}" height="{h - 2}" rx="18" fill="#ffffff" stroke="#e5e7eb" stroke-width="2"/>'
           f'{body}</svg>\n')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(doc, encoding="utf-8")


# --- Lesson 0: the three helpers ---------------------------------------------

def fig_three_helpers():
    b = ""
    b += rect(250, 22, 400, 46, YELLOW, rx=23)
    b += text(450, 51, "❓ “How long do we keep exported data?”", 16)
    b += panel(40, 110, 240, 160, BLUE, "📚 The library", ["documents cut into", "small cards,", "prepared once, in advance"])
    for i, (dx, dy, rot) in enumerate([(-30, 0, -8), (0, -4, 0), (30, 2, 8)]):
        b += f'<g transform="rotate({rot} {160 + dx} 240)">{rect(140 + dx, 222 + dy, 40, 28, ("#ffffff", BLUE[1]), rx=4)}</g>'
    b += panel(330, 110, 240, 160, GREEN, "🧑‍💼 The librarian", ["finds the 5 cards most", "likely to hold the answer,", "for every question, fast"])
    b += panel(620, 110, 240, 160, ORANGE, "✍️ The writer (AI)", ["reads only those cards,", "writes the answer and", "adds footnotes [1] [2]", "or says “I don’t know”"])
    b += arrow(450, 70, 450, 106, INK)
    b += arrow(284, 190, 326, 190, GREEN[1], both=True)
    b += text(305, 176, "search", 12, fill=GREEN[1])
    b += arrow(574, 190, 616, 190, ORANGE[1])
    b += text(595, 176, "cards", 12, fill=ORANGE[1])
    b += arrow(740, 274, 740, 300, INK)
    b += rect(560, 304, 300, 42, GREEN, rx=21)
    b += text(710, 331, "✅ “30 days [1]” – with a source", 15)
    b += text(40, 332, "RAG = look it up first, then answer.", 15, anchor="start", weight="bold", fill=MUTED)
    svg("00_three_helpers.svg", 900, 370, b, "The library, the librarian and the writer")


# --- Lesson 1: chunking ------------------------------------------------------

def fig_chunking():
    cw = 9.0  # width of one monospace character at size 15
    x0 = 40
    b = ""
    s = "use heappush to add an item while keeping the heap in order. heappop removes the smallest."
    # Row 1: fixed-size scissors
    b += text(x0, 40, "✂️  Fixed size – cut every N characters, without reading", 16, anchor="start", weight="bold")
    b += rect(x0, 54, len(s) * cw + 20, 34, GREY, rx=6)
    b += text(x0 + 10, 77, s, 15, anchor="start", mono=True)
    cuts = (s.index("keeping") + 4, s.index("heappop") + 4)
    for pos in cuts:
        x = x0 + 10 + pos * cw
        b += f'<line x1="{x}" y1="48" x2="{x}" y2="96" stroke="{RED[1]}" stroke-width="3" stroke-dasharray="5 4"/>'
    b += text(x0 + 10 + cuts[0] * cw, 112, "cuts “keeping” in half ✗", 13, fill=RED[1])
    b += text(x0 + 10 + cuts[1] * cw, 112, "cuts “heappop” in half ✗", 13, fill=RED[1])

    # Row 2: overlap like roof tiles
    y = 150
    b += text(x0, y, "🧱  Overlap – neighbouring cards share a little text (like roof tiles)", 16, anchor="start", weight="bold")
    for i, (x, colors) in enumerate([(40, BLUE), (220, GREEN), (400, ORANGE)]):
        b += f'<rect x="{x}" y="{y + 16 + i * 6}" width="230" height="40" rx="6" fill="{colors[0]}" stroke="{colors[1]}" stroke-width="2" opacity="0.85"/>'
        b += text(x + 115, y + 42 + i * 6, f"card {i + 1}", 14, weight="bold", fill=colors[1])
    b += text(245, y + 92, "shared part", 13, fill=MUTED)
    b += text(425, y + 92, "shared part", 13, fill=MUTED)
    b += text(670, y + 44, "A sentence cut at the edge", 13, anchor="start", fill=MUTED)
    b += text(670, y + 62, "of one card still appears", 13, anchor="start", fill=MUTED)
    b += text(670, y + 80, "whole in its neighbour ✓", 13, anchor="start", fill=GREEN[1])

    # Row 3: recursive at natural boundaries
    y = 275
    b += text(x0, y, "📑  Recursive – cut where the author paused: headings, then paragraphs, then sentences", 16, anchor="start", weight="bold")
    blocks = [("## heapq", 110, PURPLE), ("paragraph", 170, GREY), ("paragraph", 140, GREY),
              ("## heappush", 130, PURPLE), ("paragraph", 190, GREY)]
    x = x0
    for i, (label, w, colors) in enumerate(blocks):
        b += rect(x, y + 16, w - 6, 38, colors, rx=6)
        b += text(x + (w - 6) / 2, y + 41, label, 14, mono=label.startswith("#"))
        if i in (2, 4) or i == 0:
            pass
        x += w
    for cut in (x0 + 110 + 170 + 140 - 3,):
        b += f'<line x1="{cut}" y1="{y + 8}" x2="{cut}" y2="{y + 62}" stroke="{GREEN[1]}" stroke-width="3"/>'
    b += text(x0 + 420 - 3, y + 80, "cut before the next heading ✓", 13, fill=GREEN[1])

    # Row 4: parent-child
    y = 390
    b += text(x0, y, "🔍  Parent-child – search with a small card, give the writer its whole section", 16, anchor="start", weight="bold")
    b += rect(x0, y + 16, 560, 96, BLUE, rx=10)
    b += text(x0 + 16, y + 40, "parent = the whole “## heappush” section  →  given to the writer", 14, anchor="start", fill=BLUE[1])
    b += rect(x0 + 100, y + 56, 340, 40, GREEN, rx=8)
    b += text(x0 + 270, y + 81, "child = small card  →  used for searching", 13, fill=GREEN[1])
    b += text(630, y + 50, "Like a book index:", 14, anchor="start", weight="bold")
    b += text(630, y + 72, ["look up one precise entry,", "then read the whole page."], 14, anchor="start", fill=MUTED)
    svg("01_chunking.svg", 900, 530, b, "Four ways to cut a document into cards")


# --- Lesson 2: meaning map and cosine ---------------------------------------------

def fig_meaning_map():
    b = ""
    ox, oy = 70, 330
    b += text(40, 38, "A map of meaning: similar meaning → nearby pins", 16, anchor="start", weight="bold")
    b += arrow(ox, oy, ox, 60, MUTED)
    b += arrow(ox, oy, 470, oy, MUTED)
    b += text(ox + 8, 72, "“animal-ness”", 13, anchor="start", fill=MUTED)
    b += text(468, oy + 24, "“money-ness”", 13, anchor="end", fill=MUTED)
    pins = [("cat", 120, 110, BLUE), ("kitten", 160, 135, BLUE), ("dog", 205, 105, BLUE),
            ("stock market", 400, 270, ORANGE), ("interest rate", 360, 300, ORANGE)]
    for label, x, y, colors in pins:
        b += circle(x, y, 9, colors)
        b += text(x + 14, y - 10, label, 14, anchor="start")
    b += text(270, 200, "far apart = unrelated", 13, fill=MUTED, italic=True)

    # Right: arrows and cosine
    b += text(520, 38, "Cosine similarity: same direction?", 16, anchor="start", weight="bold")
    rows = [("same direction", 1.0, 12, 38, GREEN), ("right angle", 0.0, 0, 90, GREY), ("opposite", -1.0, 20, 200, RED)]
    for i, (label, score, a1, a2, colors) in enumerate(rows):
        cx, cy = 580, 110 + i * 95
        for ang, length in ((a1, 70), (a2, 55)):
            rad = math.radians(ang)
            b += arrow(cx, cy, cx + length * math.cos(rad), cy - length * math.sin(rad), colors[1] if colors != GREY else MUTED)
        b += circle(cx, cy, 3, (INK, INK))
        b += text(670, cy - 4, label, 15, anchor="start", weight="bold")
        b += text(670, cy + 18, f"score ≈ {score:+.0f}".replace("+0", "0"), 15, anchor="start", fill=colors[1] if colors != GREY else MUTED)
    b += text(520, 370, ["Real embeddings use 384 directions", "instead of 2 – but the idea is the same."], 13, anchor="start", fill=MUTED)
    svg("02_meaning_map.svg", 900, 400, b, "Embeddings place text on a map of meaning")


# --- Lesson 3: vector search ------------------------------------------------------

def fig_vector_search():
    b = ""
    b += text(40, 38, "Vector search: find the card pins closest to the question", 16, anchor="start", weight="bold")
    b += rect(40, 56, 470, 300, GREY, rx=14, width=1.5)
    qx, qy = 260, 200
    near = [(205, 160, "#1  0.82", "middle", 0, -14), (320, 165, "#2  0.79", "middle", 0, -14),
            (235, 262, "#3  0.75", "start", 18, 5)]
    far = [(90, 90), (130, 310), (430, 90), (470, 300), (400, 250), (100, 200), (460, 170), (170, 100), (380, 320), (330, 85)]
    for x, y in far:
        b += f'<rect x="{x - 12}" y="{y - 8}" width="24" height="16" rx="3" fill="#ffffff" stroke="{GREY[1]}" stroke-width="1.5"/>'
    b += circle(qx, qy, 95, ("none", GREEN[1]), dash=True)
    for x, y, label, anchor, dx, dy in near:
        b += f'<rect x="{x - 12}" y="{y - 8}" width="24" height="16" rx="3" fill="{GREEN[0]}" stroke="{GREEN[1]}" stroke-width="2"/>'
        b += text(x + dx, y + dy, label, 12, anchor=anchor, fill=GREEN[1], weight="bold", halo=True)
    b += text(qx, qy + 9, "★", 26, fill=ORANGE[1])
    b += text(qx + 18, qy + 5, "your question", 12, anchor="start", fill=ORANGE[1], weight="bold", halo=True)
    b += text(445, 345, "grey = ignored", 12, fill=MUTED)
    steps = [("1", "Turn every card into numbers", "(once, ahead of time)"),
             ("2", "Turn the question into numbers", "(for each question)"),
             ("3", "Score every card against it", "(cosine similarity)"),
             ("4", "Keep the top k", "(here k = 3)")]
    for i, (n, line1, line2) in enumerate(steps):
        y = 90 + i * 70
        b += circle(565, y, 17, GREEN)
        b += text(565, y + 6, n, 16, weight="bold", fill=GREEN[1])
        b += text(595, y, line1, 15, anchor="start", weight="bold")
        b += text(595, y + 20, line2, 13, anchor="start", fill=MUTED)
    svg("03_vector_search.svg", 900, 380, b, "Vector search keeps the k nearest cards")


# --- Lesson 4: BM25 ---------------------------------------------------------------

def fig_bm25():
    b = ""
    b += text(40, 38, "Idea 1 · Rare words are better clues (IDF)", 16, anchor="start", weight="bold")
    b += text(40, 60, "100 cards in the library", 13, anchor="start", fill=MUTED)
    words = [("“return”", "on 90 cards", 0.11, GREY), ("“directory”", "on 10 cards", 2.26, YELLOW), ("“rmtree”", "on 1 card", 4.21, GREEN)]
    for i, (w, on, idf, colors) in enumerate(words):
        y = 90 + i * 62
        b += text(40, y + 20, w, 15, anchor="start", weight="bold")
        b += text(40, y + 38, on, 12, anchor="start", fill=MUTED)
        b += rect(150, y + 4, max(6, idf * 62), 30, colors, rx=5)
        b += text(150 + max(6, idf * 62) + 8, y + 25, f"{idf:.2f}", 14, anchor="start", weight="bold")
    b += text(40, 298, ["“two eyes” describes everyone;", "“green hair” finds your friend."], 13, anchor="start", fill=MUTED, italic=True)

    b += text(480, 38, "Idea 2 · Repeats help, but level off", 16, anchor="start", weight="bold")
    gx, gy, gw, gh = 500, 270, 340, 190
    b += arrow(gx, gy, gx, gy - gh - 10, MUTED)
    b += arrow(gx, gy, gx + gw + 10, gy, MUTED)
    pts_bm25 = " ".join(f"{gx + tf * gw / 20:.1f},{gy - (tf * 2.5 / (tf + 1.5)) * gh / 3.2:.1f}" for tf in range(21))
    pts_lin = " ".join(f"{gx + tf * gw / 20:.1f},{gy - (tf / 6.5) * gh / 3.2:.1f}" for tf in range(0, 21))
    b += f'<polyline points="{pts_lin}" fill="none" stroke="{GREY[1]}" stroke-width="2.5" stroke-dasharray="6 5"/>'
    b += f'<polyline points="{pts_bm25}" fill="none" stroke="{GREEN[1]}" stroke-width="3.5"/>'
    b += text(gx + gw - 4, gy - gh + 4, "plain counting", 13, anchor="end", fill=MUTED)
    b += text(gx + 170, gy - 143, "BM25", 14, weight="bold", fill=GREEN[1], halo=True)
    b += text(gx + gw / 2, gy + 24, "times the word appears on the card", 13, fill=MUTED)
    b += text(gx - 10, gy - gh / 2, "points", 13, anchor="end", fill=MUTED)
    b += text(480, 322, ["A review saying “pizza” 3 times is about pizza.", "Saying it 30 times isn’t 10× more about pizza."], 13, anchor="start", fill=MUTED, italic=True)
    svg("04_bm25.svg", 900, 365, b, "BM25: rare words count more and repeats level off")


# --- Lesson 5: RRF ------------------------------------------------------------------

def fig_rrf():
    b = ""
    b += text(40, 38, "Combine the two lists by position, not by score", 16, anchor="start", weight="bold")
    lists = [("🔤 Keyword librarian", 40, BLUE, ["A", "B", "D"]), ("💡 Meaning librarian", 40, PURPLE, ["C", "E", "A"])]
    for j, (title, x, colors, items) in enumerate(lists):
        y = 70 + j * 150
        b += rect(x, y, 230, 130, colors, rx=12)
        b += text(x + 115, y + 28, title, 15, weight="bold", fill=colors[1])
        for i, it in enumerate(items):
            hl = it == "A"
            b += text(x + 70, y + 60 + i * 24, f"{i + 1}.", 15, anchor="end", fill=MUTED)
            b += text(x + 82, y + 60 + i * 24, f"card {it}", 15, anchor="start", weight="bold" if hl else "normal",
                      fill=GREEN[1] if hl else INK)
    b += arrow(275, 135, 330, 175, MUTED)
    b += arrow(275, 285, 330, 225, MUTED)
    b += rect(335, 80, 330, 220, GREY, rx=12)
    b += text(500, 108, "Each list gives 1 ÷ (60 + position)", 15, weight="bold")
    rows = [("A", "1/61 + 1/63", "0.0323", True), ("C", "1/61", "0.0164", False), ("B", "1/62", "0.0161", False),
            ("E", "1/62", "0.0161", False), ("D", "1/63", "0.0159", False)]
    for i, (c, f, v, hl) in enumerate(rows):
        y = 142 + i * 32
        col = GREEN[1] if hl else INK
        b += text(360, y, f"card {c}", 15, anchor="start", weight="bold" if hl else "normal", fill=col)
        b += text(450, y, f, 15, anchor="start", fill=col, mono=True)
        b += text(645, y, v, 15, anchor="end", weight="bold" if hl else "normal", fill=col, mono=True)
    b += arrow(670, 190, 715, 190, GREEN[1])
    b += rect(720, 110, 145, 160, GREEN, rx=12)
    b += text(792, 138, "Combined", 15, weight="bold", fill=GREEN[1])
    for i, c in enumerate(["A", "C", "B"]):
        b += text(760, 172 + i * 28, f"{i + 1}. card {c}", 15, anchor="start", weight="bold" if c == "A" else "normal")
    b += text(40, 382, "Card A wins: both librarians liked it. Agreement between two different methods is a strong signal.",
              14, anchor="start", fill=MUTED, italic=True)
    svg("05_rrf.svg", 900, 405, b, "Reciprocal Rank Fusion combines two ranked lists")


# --- Lesson 6: re-ranking ------------------------------------------------------------

def fig_rerank():
    b = ""
    b += text(40, 38, "Two stages: a quick screening, then a careful look", 16, anchor="start", weight="bold")
    b += rect(40, 70, 190, 170, BLUE, rx=14)
    b += text(135, 100, "📚 all cards", 16, weight="bold", fill=BLUE[1])
    b += text(135, 125, "hundreds", 15)
    for i in range(18):
        x, y = 62 + (i % 6) * 26, 148 + (i // 6) * 26
        b += f'<rect x="{x}" y="{y}" width="20" height="14" rx="2" fill="#ffffff" stroke="{BLUE[1]}" stroke-width="1.2"/>'
    b += arrow(234, 155, 330, 155, GREEN[1])
    b += text(282, 118, ["quick", "screening"], 13, fill=GREEN[1])
    b += text(282, 182, "milliseconds", 12, fill=MUTED)
    b += rect(335, 95, 160, 120, GREEN, rx=14)
    b += text(415, 125, "30 candidates", 15, weight="bold", fill=GREEN[1])
    for i in range(8):
        x, y = 357 + (i % 4) * 30, 145 + (i // 4) * 26
        b += f'<rect x="{x}" y="{y}" width="22" height="15" rx="2" fill="#ffffff" stroke="{GREEN[1]}" stroke-width="1.2"/>'
    b += arrow(499, 155, 600, 155, ORANGE[1])
    b += text(550, 118, ["careful", "reader"], 13, fill=ORANGE[1])
    b += text(550, 182, "≈ 1.5 seconds", 12, fill=MUTED)
    b += rect(605, 110, 120, 90, ORANGE, rx=14)
    b += text(665, 140, "best 5", 15, weight="bold", fill=ORANGE[1])
    b += text(665, 170, "🏆", 22)
    b += text(745, 150, ["Like 1,000 CVs →", "30 interviews →", "5 hires"], 13, anchor="start", fill=MUTED, italic=True)

    y = 270
    b += text(40, y, "Why the careful reader is smarter", 16, anchor="start", weight="bold")
    b += rect(40, y + 16, 400, 120, GREY, rx=12)
    b += text(240, y + 42, "Fast librarian: read separately", 14, weight="bold")
    b += rect(70, y + 60, 100, 30, YELLOW, rx=6)
    b += text(120, y + 80, "question", 13)
    b += rect(70, y + 98, 100, 30, BLUE, rx=6)
    b += text(120, y + 118, "card", 13)
    b += arrow(174, y + 75, 240, y + 75, MUTED)
    b += arrow(174, y + 113, 240, y + 113, MUTED)
    b += text(285, y + 80, "[0.2, …]", 13, mono=True)
    b += text(285, y + 118, "[0.3, …]", 13, mono=True)
    b += text(390, y + 100, "compare", 13, fill=MUTED)
    b += rect(470, y + 16, 400, 120, GREY, rx=12)
    b += text(670, y + 42, "Careful reader: reads both together", 14, weight="bold")
    b += rect(500, y + 60, 200, 66, ORANGE, rx=8, dash=True)
    b += rect(512, y + 68, 176, 22, YELLOW, rx=5)
    b += text(600, y + 84, "question", 13)
    b += rect(512, y + 96, 176, 22, BLUE, rx=5)
    b += text(600, y + 112, "card", 13)
    b += arrow(704, y + 93, 760, y + 93, ORANGE[1])
    b += text(805, y + 98, "score", 14, weight="bold", fill=ORANGE[1])
    svg("06_rerank.svg", 900, 420, b, "Re-ranking: quick screening then careful reading")


# --- Lesson 7: the prompt -------------------------------------------------------------

def fig_prompt():
    b = ""
    b += text(40, 38, "What the writer actually receives – and nothing else", 16, anchor="start", weight="bold")
    b += rect(40, 58, 380, 330, ("#ffffff", INK), rx=12, width=1.5)
    b += rect(56, 72, 348, 120, PURPLE, rx=8)
    b += text(72, 96, "RULES (system prompt)", 14, anchor="start", weight="bold", fill=PURPLE[1])
    b += text(72, 120, ["• use ONLY the numbered cards", "• add a footnote [n] to every fact",
                        "• not in the cards? say “I don’t know”"], 13, anchor="start")
    b += rect(56, 202, 348, 120, BLUE, rx=8)
    b += text(72, 226, "CARDS from the librarian", 14, anchor="start", weight="bold", fill=BLUE[1])
    for i, t in enumerate(["random.sample(population, k) …", "random.choice(seq) …", "random.shuffle(x) …"]):
        b += text(72, 252 + i * 22, f"[{i + 1}] {t}", 13, anchor="start", mono=True)
    b += rect(56, 332, 348, 44, YELLOW, rx=8)
    b += text(72, 360, "QUESTION: pick 3 different items from a list?", 13, anchor="start")
    b += arrow(424, 220, 490, 220, ORANGE[1])
    b += rect(495, 180, 150, 80, ORANGE, rx=14)
    b += text(570, 214, "✍️ writer", 16, weight="bold", fill=ORANGE[1])
    b += text(570, 238, "(AI)", 13, fill=MUTED)
    b += arrow(649, 220, 690, 220, ORANGE[1])
    b += rect(695, 180, 175, 80, GREEN, rx=14)
    b += text(782, 210, "“Use random.sample", 13)
    b += text(782, 230, "with k=3 [1]”", 13)
    b += path("M 782 264 C 782 300, 600 310, 410 254", GREEN[1], dash=True)
    b += text(600, 316, "the footnote lets you check the source", 12, fill=GREEN[1])

    b += text(490, 352, "Three classic traps", 14, anchor="start", weight="bold")
    traps = [("🗂️", "old versions"), ("🤷", "no answer in cards"), ("🕵️", "hidden instructions")]
    for i, (icon, label) in enumerate(traps):
        y = 366 + i * 38
        b += rect(490, y, 230, 30, RED, rx=15)
        b += text(605, y + 20, f"{icon}  {label}", 13, fill=RED[1])
    svg("07_prompt.svg", 900, 490, b, "The writer receives rules, numbered cards and the question")


# --- Lesson 8: evaluation --------------------------------------------------------------

def fig_eval():
    b = ""
    b += text(40, 38, "One exam question, scored two ways", 16, anchor="start", weight="bold")
    b += rect(40, 58, 260, 150, YELLOW, rx=12)
    b += text(56, 86, "📝 Exam question", 14, anchor="start", weight="bold", fill=YELLOW[1])
    b += text(56, 112, ["“How can I split a URL", "into its parts?”"], 14, anchor="start")
    b += text(56, 168, "Answer key:", 13, anchor="start", fill=MUTED)
    b += text(56, 190, "py.urllib.parse", 14, anchor="start", weight="bold", mono=True)
    b += arrow(304, 133, 345, 133, MUTED)
    b += rect(350, 58, 230, 190, GREY, rx=12)
    b += text(465, 84, "Librarian’s top 5", 14, weight="bold")
    found = ["py.shutil", "py.urllib.parse", "py.re", "py.csv", "py.string"]
    for i, d in enumerate(found):
        y = 112 + i * 26
        ok = d == "py.urllib.parse"
        b += text(372, y, f"{i + 1}.", 14, anchor="start", fill=MUTED)
        b += text(395, y, d, 14, anchor="start", mono=True, weight="bold" if ok else "normal", fill=GREEN[1] if ok else INK)
        if ok:
            b += text(560, y, "✓", 16, fill=GREEN[1], weight="bold")
    b += arrow(584, 120, 625, 100, MUTED)
    b += arrow(584, 170, 625, 190, MUTED)
    b += rect(630, 62, 240, 76, GREEN, rx=12)
    b += text(750, 90, "Recall@5 = 1", 16, weight="bold", fill=GREEN[1])
    b += text(750, 114, "“it’s in the top 5”", 13, fill=MUTED)
    b += rect(630, 152, 240, 76, ORANGE, rx=12)
    b += text(750, 180, "Reciprocal rank = ½", 16, weight="bold", fill=ORANGE[1])
    b += text(750, 204, "“it came 2nd”", 13, fill=MUTED)

    b += text(40, 290, "Don’t peek at the exam", 16, anchor="start", weight="bold")
    b += rect(40, 306, 400, 60, BLUE, rx=12)
    b += text(240, 332, "🏋️ Dev set = practice exam", 15, weight="bold", fill=BLUE[1])
    b += text(240, 354, "use it as often as you like while improving", 13, fill=MUTED)
    b += rect(470, 306, 400, 60, RED, rx=12)
    b += text(670, 332, "🔒 Test set = final exam", 15, weight="bold", fill=RED[1])
    b += text(670, 354, "keep it locked; use it once, at the end", 13, fill=MUTED)
    svg("08_eval.svg", 900, 390, b, "Scoring retrieval with Recall@k and reciprocal rank")


# --- Lesson 9: query transformations --------------------------------------------------------

def fig_query():
    b = ""
    b += text(40, 38, "Multi-Query: ask the question several ways, combine the results", 16, anchor="start", weight="bold")
    b += rect(40, 70, 190, 70, YELLOW, rx=12)
    b += text(135, 98, ["“make my script", "stop for a bit”"], 14)
    rewrites = ["time.sleep pause script", "delay execution seconds", "suspend program briefly"]
    for i, r in enumerate(rewrites):
        y = 58 + i * 36
        b += arrow(234, 105, 280, y + 13, MUTED)
        b += rect(285, y, 230, 26, PURPLE, rx=13)
        b += text(400, y + 18, r, 13, fill=PURPLE[1])
        b += arrow(519, y + 13, 560, y + 13, GREEN[1])
        b += rect(565, y, 90, 26, GREEN, rx=6)
        b += text(610, y + 18, "list " + "ABC"[i], 13, fill=GREEN[1])
    b += arrow(659, 107, 705, 107, GREEN[1])
    b += rect(710, 80, 160, 54, GREEN, rx=12)
    b += text(790, 104, "RRF (Lesson 5)", 14, weight="bold", fill=GREEN[1])
    b += text(790, 124, "→ best cards", 13)

    y0 = 200
    b += text(40, y0, "HyDE: imagine an answer, then find real cards that look like it", 16, anchor="start", weight="bold")
    b += rect(40, y0 + 26, 190, 70, YELLOW, rx=12)
    b += text(135, y0 + 54, ["“make my script", "stop for a bit”"], 14)
    b += arrow(234, y0 + 61, 280, y0 + 61, ORANGE[1])
    b += rect(285, y0 + 18, 240, 88, ORANGE, rx=8, dash=True)
    b += text(405, y0 + 42, "✍️ pretend answer", 14, weight="bold", fill=ORANGE[1])
    b += text(405, y0 + 66, ["“Suspend execution for the", "given number of seconds…”"], 13, italic=True)
    b += arrow(529, y0 + 61, 575, y0 + 61, GREEN[1])
    b += text(552, y0 + 46, "search", 12, fill=GREEN[1])
    b += rect(580, y0 + 26, 290, 70, GREEN, rx=12)
    b += text(725, y0 + 54, "real card: time.sleep(secs) ✓", 14, weight="bold", fill=GREEN[1])
    b += text(725, y0 + 78, "only real cards are used", 13, fill=MUTED)
    b += text(405, y0 + 128, "the pretend answer is never shown to the user", 13, fill=MUTED, italic=True)
    b += text(40, y0 + 168, "Like finding a lost dog by showing a photo of a similar-looking one.", 13, anchor="start",
              fill=MUTED, italic=True)
    svg("09_query.svg", 900, 390, b, "Multi-Query and HyDE improve the question before searching")


# --- Lesson 10: check-and-retry and agents -----------------------------------------------------

def fig_crag():
    b = ""
    b += text(40, 38, "Corrective RAG: check the cards before writing", 16, anchor="start", weight="bold")
    b += rect(40, 60, 150, 44, YELLOW, rx=22)
    b += text(115, 88, "❓ question", 14)
    b += arrow(115, 108, 115, 134, MUTED)
    b += rect(40, 138, 150, 50, GREEN, rx=10)
    b += text(115, 168, "🧑‍💼 librarian", 14, weight="bold", fill=GREEN[1])
    b += arrow(194, 163, 238, 163, MUTED)
    b += f'<polygon points="320,118 400,163 320,208 240,163" fill="{PURPLE[0]}" stroke="{PURPLE[1]}" stroke-width="2"/>'
    b += text(320, 160, "any card", 13, weight="bold", fill=PURPLE[1])
    b += text(320, 176, "helpful?", 13, weight="bold", fill=PURPLE[1])
    b += arrow(404, 163, 450, 163, GREEN[1])
    b += text(427, 150, "yes", 13, weight="bold", fill=GREEN[1])
    b += rect(455, 138, 150, 50, ORANGE, rx=10)
    b += text(530, 168, "✍️ write answer", 14, weight="bold", fill=ORANGE[1])
    b += arrow(320, 212, 320, 250, RED[1])
    b += text(334, 236, "no", 13, anchor="start", weight="bold", fill=RED[1])
    b += rect(235, 254, 170, 50, RED, rx=10)
    b += text(320, 284, "✏️ rewrite question", 14, fill=RED[1])
    b += path("M 235 279 C 140 279, 115 250, 115 192", RED[1])
    b += text(150, 300, "search again", 12, fill=RED[1])
    b += rect(455, 254, 150, 50, GREY, rx=10)
    b += text(530, 284, "🤷 “I don’t know”", 14)
    b += arrow(409, 279, 451, 279, MUTED, dash=True)
    b += text(430, 320, "still nothing on the 2nd try", 12, fill=MUTED)

    b += text(760, 38, "An agent decides for itself", 16, weight="bold")
    b += text(760, 64, "like a detective following clues", 12, fill=MUTED, italic=True)
    pills = {"search": (840, 130, 100, "🔍 search"), "read": (680, 130, 126, "📖 read results"),
             "think": (760, 250, 100, "🤔 think")}
    for x, y, w, label in pills.values():
        b += rect(x - w / 2, y - 18, w, 36, BLUE, rx=18)
        b += text(x, y + 5, label, 13, fill=BLUE[1], weight="bold")
    b += arrow(785, 232, 825, 152, INK)
    b += arrow(788, 130, 747, 130, INK)
    b += arrow(700, 150, 736, 230, INK)
    b += arrow(760, 270, 760, 306, GREEN[1])
    b += rect(670, 310, 180, 36, GREEN, rx=18)
    b += text(760, 334, "✅ answer when ready", 13, weight="bold", fill=GREEN[1])
    svg("10_crag.svg", 900, 400, b, "Corrective RAG and the agent loop")


# --- Lesson 11: the NLP map ----------------------------------------------------------

def fig_nlp_map():
    b = ""
    b += text(40, 38, "NLP: one kitchen, many dishes", 16, anchor="start", weight="bold")
    b += rect(450, 62, 420, 178, YELLOW, rx=16, dash=True)
    b += text(660, 228, "RAG = Search + Generate (this course)", 14, weight="bold", fill=YELLOW[1])
    families = [("🗂️ Classify", ["spam or not?", "good or bad review?", "which topic?"], "Lesson 11", PURPLE, 40),
                ("🔎 Extract", ["names, dates,", "amounts in a", "contract"], "Lesson 12 path", GREY, 245),
                ("🧑‍💼 Search", ["find the text", "that matches", "a question"], "Lessons 2-6", GREEN, 465),
                ("✍️ Generate", ["translate, summarise,", "answer, chat"], "Lesson 7", ORANGE, 670)]
    for title, lines, where, colors, x in families:
        b += rect(x, 78, 190, 126, colors, rx=12)
        b += text(x + 95, 104, title, 15, weight="bold", fill=colors[1])
        b += text(x + 95, 128, lines, 13)
        b += text(x + 95, 194, where, 12, fill=MUTED, italic=True)
    b += path("M 135 252 L 135 266 L 765 266 L 765 252", MUTED, end=False, width=2)
    b += path("M 450 266 L 450 286", MUTED)
    b += text(470, 282, "all built from the same ingredients", 12, anchor="start", fill=MUTED, italic=True)
    ingredients = [("tokens", "split text into pieces", "Lessons 4, 12"),
                   ("embeddings", "meaning as numbers", "Lesson 2"),
                   ("Transformers / LLMs", "models that read and write", "Lessons 6, 7"),
                   ("evaluation", "fair tests, read failures", "Lessons 8, 11")]
    for i, (name, what, where) in enumerate(ingredients):
        x = 40 + i * 210
        b += rect(x, 296, 190, 84, BLUE, rx=12)
        b += text(x + 95, 322, name, 15, weight="bold", fill=BLUE[1])
        b += text(x + 95, 344, what, 13)
        b += text(x + 95, 366, where, 12, fill=MUTED, italic=True)
    svg("11_nlp_map.svg", 900, 400, b, "A map of NLP: task families built from shared ingredients")


# --- Lesson 12: the learning path ----------------------------------------------------------

def fig_learning_path():
    b = ""
    b += text(40, 38, "From driver to engineer: five stages", 16, anchor="start", weight="bold")
    b += f'<line x1="70" y1="120" x2="860" y2="120" stroke="{LINE}" stroke-width="10" stroke-linecap="round"/>'
    b += f'<line x1="70" y1="120" x2="860" y2="120" stroke="#ffffff" stroke-width="2" stroke-dasharray="14 12"/>'
    b += text(70, 84, "🚗 you are here: driver", 13, anchor="start", weight="bold", fill=GREEN[1])
    b += text(380 + 80, 84, "🔧 mechanic", 13, weight="bold", fill=PURPLE[1])
    b += text(550 + 80, 84, "🏭 engineer", 13, weight="bold", fill=ORANGE[1])
    stages = [("Text as data", ["Yandex NLP, week 1", "nlp-tutorial", "sentiment-analysis"], "≈ 2 weeks", BLUE),
              ("How networks learn", ["Karpathy:", "Zero to Hero"], "≈ 3 weeks", PURPLE),
              ("Transformers", ["Hugging Face course", "ch. 1-3 and 7"], "≈ 3 weeks", GREEN),
              ("Build a small GPT", ["LLMs from Scratch", "nanoGPT"], "≈ 4 weeks", ORANGE),
              ("LLM applications", ["mlabonne llm-course", "Yandex week 9 (RAG)"], "ongoing", YELLOW)]
    for i, (title, lines, time, colors) in enumerate(stages):
        x = 40 + i * 170
        cx = x + 80
        b += circle(cx, 120, 20, colors, width=3)
        b += text(cx, 127, str(i + 1), 17, weight="bold", fill=colors[1])
        b += rect(x, 152, 160, 170, colors, rx=12)
        b += text(cx, 180, title, 14, weight="bold", fill=colors[1])
        b += text(cx, 212, lines, 13)
        b += text(cx, 306, time, 12, fill=MUTED, italic=True)
    b += text(40, 356, "All five are free on GitHub and already downloaded into the references folder.",
              13, anchor="start", fill=MUTED, italic=True)
    svg("12_learning_path.svg", 900, 380, b, "A five-stage NLP learning path")


if __name__ == "__main__":
    for fig in (fig_three_helpers, fig_chunking, fig_meaning_map, fig_vector_search, fig_bm25, fig_rrf,
                fig_rerank, fig_prompt, fig_eval, fig_query, fig_crag, fig_nlp_map, fig_learning_path):
        fig()
    print("wrote", len(list(OUT.glob("*.svg"))), "figures to", OUT)
