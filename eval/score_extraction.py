from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Tuple


def _norm(text: str) -> str:
    """Normalize text for comparison."""
    return unicodedata.normalize("NFKC", text).strip()


def levenshtein(a: str, b: str) -> int:
    """Classic dynamic programming edit distance."""
    a, b = _norm(a), _norm(b)
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[-1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def normalized_distance(a: str, b: str) -> float:
    denom = max(len(_norm(a)), len(_norm(b)), 1)
    return levenshtein(a, b) / denom


def diacritic_preservation(a: str, b: str) -> float:
    """Share of combining-mark code points preserved."""

    def marks(s: str) -> List[Tuple[int, str]]:
        return [
            (i, c)
            for i, c in enumerate(_norm(s))
            if unicodedata.combining(c)
        ]

    ma = set(marks(a))
    mb = set(marks(b))
    if not ma and not mb:
        return 1.0
    return len(ma & mb) / max(len(ma | mb), 1)


@dataclass
class Scores:
    n: int
    token_acc: float
    char_dist: float
    diacritics: float


def score_pairs(pred: List[Dict], truth: List[Dict]) -> Scores:
    assert len(pred) == len(truth), "pred/truth length mismatch"
    n = len(truth)
    tok_hits = 0
    char_dists: List[float] = []
    diacs: List[float] = []
    for p, t in zip(pred, truth):
        pe, pd = _norm(p["english"]), _norm(p["dakota"])
        te, td = _norm(t["english"]), _norm(t["dakota"])
        tok_hits += int(pe == te and pd == td)
        char_dists.append(
            (normalized_distance(pe, te) + normalized_distance(pd, td)) / 2
        )
        diacs.append(
            (diacritic_preservation(pe, te) + diacritic_preservation(pd, td)) / 2
        )
    return Scores(
        n=n,
        token_acc=tok_hits / max(n, 1),
        char_dist=sum(char_dists) / max(n, 1),
        diacritics=sum(diacs) / max(n, 1),
    )


def load_pairs(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
