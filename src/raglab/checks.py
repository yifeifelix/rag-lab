"""Auto-checkers for the exercises.

Each ``check_*`` function runs your implementation on a few test cases and
prints ✅ or ❌ with a hint. They never reveal the reference solution.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


def _report(name: str, failures: list[str], passed: int) -> bool:
    if failures:
        print(f"❌ {name}: {len(failures)} of {passed + len(failures)} checks failed")
        for msg in failures[:3]:
            print("   •", msg)
        return False
    print(f"✅ {name}: all {passed} checks passed - nice work!")
    return True


def _run(name: str, cases: list[tuple[str, Callable[[], bool], str]]) -> bool:
    failures, passed = [], 0
    for label, test, hint in cases:
        try:
            ok = bool(test())
        except NotImplementedError:
            print(f"⏳ {name}: not implemented yet - replace `raise NotImplementedError` with your code")
            return False
        except Exception as exc:  # show learner errors as hints, not tracebacks
            ok, hint = False, f"{hint} (raised {type(exc).__name__}: {exc})"
        if ok:
            passed += 1
        else:
            failures.append(f"{label}: {hint}")
    return _report(name, failures, passed)


# Lesson 1 -----------------------------------------------------------------

def check_fixed_size_chunker(fn: Callable[[str, int, int], list[str]]) -> bool:
    text = "abcdefghijklmnopqrstuvwxyz"
    return _run("fixed_size_chunker", [
        ("no overlap", lambda: fn(text, 10, 0) == ["abcdefghij", "klmnopqrst", "uvwxyz"],
         "size=10, overlap=0 should give 3 pieces of up to 10 chars"),
        ("with overlap", lambda: fn(text, 10, 4)[:2] == ["abcdefghij", "ghijklmnop"],
         "the 2nd window should start `size - overlap` characters after the 1st"),
        ("covers the end", lambda: fn(text, 10, 4)[-1].endswith("z"),
         "the last window must reach the end of the text"),
        ("no tiny duplicate tail", lambda: len(fn(text, 10, 4)) == 4,
         "stop once a window reaches the end of the text (expected 4 windows)"),
        ("short text", lambda: fn("hello", 10, 2) == ["hello"], "text shorter than size is one chunk"),
    ])


# Lesson 2 -----------------------------------------------------------------

def check_cosine_similarity(fn: Callable[[np.ndarray, np.ndarray], float]) -> bool:
    a, b = np.array([1.0, 0.0]), np.array([0.0, 2.0])
    c = np.array([3.0, 4.0])
    return _run("cosine_similarity", [
        ("identical direction", lambda: math.isclose(fn(c, c * 10), 1.0, rel_tol=1e-5), "scaled copies should score 1.0"),
        ("orthogonal", lambda: math.isclose(fn(a, b), 0.0, abs_tol=1e-6), "perpendicular vectors should score 0.0"),
        ("opposite", lambda: math.isclose(fn(c, -c), -1.0, rel_tol=1e-5), "opposite vectors should score -1.0"),
        ("general", lambda: math.isclose(fn(a, c), 0.6, rel_tol=1e-5), "[1,0] vs [3,4] should be 0.6 - divide by both lengths"),
    ])


# Lesson 3 -----------------------------------------------------------------

def check_top_k(fn: Callable[[np.ndarray, int], list[int]]) -> bool:
    scores = np.array([0.1, 0.9, 0.3, 0.7, 0.5])
    return _run("top_k", [
        ("order", lambda: list(fn(scores, 3)) == [1, 3, 4], "return indices of the 3 highest scores, best first"),
        ("k larger than n", lambda: list(fn(scores, 10)) == [1, 3, 4, 2, 0], "k > len(scores) should return all indices"),
        ("k = 1", lambda: list(fn(scores, 1)) == [1], "k=1 returns only the best index"),
    ])


# Lesson 4 -----------------------------------------------------------------

def check_idf(fn: Callable[[int, int], float]) -> bool:
    ref = lambda n, df: math.log(1 + (n - df + 0.5) / (df + 0.5))
    return _run("idf", [
        ("rare term", lambda: math.isclose(fn(100, 1), ref(100, 1), rel_tol=1e-6), "use log(1 + (N - df + 0.5) / (df + 0.5))"),
        ("common term", lambda: math.isclose(fn(100, 90), ref(100, 90), rel_tol=1e-6), "common terms should get a small idf"),
        ("rare > common", lambda: fn(100, 1) > fn(100, 50) > 0, "idf must shrink as df grows and stay positive"),
    ])


def check_bm25_term(fn: Callable[..., float]) -> bool:
    def ref(tf, idf, doc_len, avg_len, k1=1.5, b=0.75):
        return idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avg_len))
    cases = [(1, 2.0, 100, 100), (5, 2.0, 100, 100), (1, 2.0, 400, 100), (0, 2.0, 50, 100)]
    return _run("bm25_term_score", [
        (f"tf={tf}, len={dl}", (lambda tf=tf, idf=idf, dl=dl, al=al:
                                math.isclose(fn(tf, idf, dl, al), ref(tf, idf, dl, al), rel_tol=1e-6)),
         "check the length normalisation: 1 - b + b * doc_len / avg_len") for tf, idf, dl, al in cases
    ] + [("saturation", lambda: fn(20, 1.0, 100, 100) < 2.6, "tf should saturate: the score can't exceed idf * (k1 + 1)")])


# Lesson 5 -----------------------------------------------------------------

def check_rrf(fn: Callable[..., dict]) -> bool:
    r1, r2 = ["a", "b", "c"], ["c", "a", "d"]
    def ref(rankings, k=60):
        out = {}
        for ranking in rankings:
            for rank, item in enumerate(ranking, 1):
                out[item] = out.get(item, 0) + 1 / (k + rank)
        return out
    def same(x, y):
        return set(x) == set(y) and all(math.isclose(x[i], y[i], rel_tol=1e-9) for i in x)
    return _run("reciprocal_rank_fusion", [
        ("two lists", lambda: same(fn([r1, r2]), ref([r1, r2])), "each list adds 1 / (k + rank), ranks start at 1"),
        ("custom k", lambda: same(fn([r1, r2], k=1), ref([r1, r2], 1)), "use the k argument"),
        ("winner", lambda: max(fn([r1, r2]), key=fn([r1, r2]).get) == "a", "'a' (ranks 1 and 2) should win"),
    ])


# Lesson 8 -----------------------------------------------------------------

def check_recall_at_k(fn: Callable[..., float]) -> bool:
    return _run("recall_at_k", [
        ("hit", lambda: fn(["x", "a", "y"], ["a"], 3) == 1.0, "relevant doc is in the top 3"),
        ("cut-off", lambda: fn(["x", "y", "a"], ["a"], 2) == 0.0, "only look at the first k results"),
        ("partial", lambda: fn(["a", "x", "y"], ["a", "b"], 3) == 0.5, "1 of 2 relevant docs found -> 0.5"),
    ])


def check_mrr(fn: Callable[..., float]) -> bool:
    return _run("reciprocal_rank", [
        ("first", lambda: fn(["a", "b"], ["a"]) == 1.0, "relevant at rank 1 -> 1.0"),
        ("third", lambda: math.isclose(fn(["x", "y", "a"], ["a"]), 1 / 3), "relevant at rank 3 -> 1/3"),
        ("missing", lambda: fn(["x", "y"], ["a"]) == 0.0, "no relevant doc -> 0.0"),
        ("first of many", lambda: fn(["x", "b", "a"], ["a", "b"]) == 0.5, "use the FIRST relevant doc"),
    ])
