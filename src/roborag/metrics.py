"""Retrieval and answer metrics. Pure functions on lists, so they are easy to unit-test.

Retrieval (per question, `rels` = relevance of the ranked results, best first):
    hit@k      1 if any of the top-k results is relevant (a.k.a. success@k / recall@k
               when each question needs one supporting passage)
    MRR@k      1 / rank of the first relevant result (0 if none in the top k)
    nDCG@k     DCG of the ranking / DCG of the ideal ranking, binary gains
    P@k        fraction of the top k that is relevant (how much noise the LLM must read)
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Sequence

CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def hit_at_k(rels: Sequence[bool], k: int) -> float:
    return float(any(rels[:k]))


def mrr_at_k(rels: Sequence[bool], k: int) -> float:
    for i, r in enumerate(rels[:k], start=1):
        if r:
            return 1.0 / i
    return 0.0


def precision_at_k(rels: Sequence[bool], k: int) -> float:
    return sum(rels[:k]) / k if k else 0.0


def ndcg_at_k(rels: Sequence[bool], k: int, n_relevant: int | None = None) -> float:
    """Binary-gain nDCG. `n_relevant` = relevant items that exist (for the ideal ranking);
    defaults to the number of relevant items in the list itself."""
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(rels[:k], start=1) if r)
    ideal_n = min(k, n_relevant if n_relevant is not None else sum(rels))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg else 0.0


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean (questions resampled with replacement)."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def paired_bootstrap_p(a: Sequence[float], b: Sequence[float], n_boot: int = 5000,
                       seed: int = 0) -> float:
    """Two-sided p-value that mean(a) == mean(b), resampling questions (paired)."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    obs = sum(diffs) / len(diffs)
    centred = [d - obs for d in diffs]
    rng = random.Random(seed)
    n = len(diffs)
    extreme = sum(abs(sum(rng.choice(centred) for _ in range(n)) / n) >= abs(obs) - 1e-12
                  for _ in range(n_boot))
    return extreme / n_boot


def parse_citations(answer: str) -> list[int]:
    """'... uses DDS [1][3] and [2, 4].' -> [1, 3, 2, 4] (order of first appearance)."""
    seen: list[int] = []
    for m in CITATION_RE.finditer(answer):
        for n in m.group(1).split(","):
            v = int(n)
            if v not in seen:
                seen.append(v)
    return seen


def summarize(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    lo, hi = bootstrap_ci(values)
    return {"mean": sum(values) / len(values), "ci_low": lo, "ci_high": hi}
