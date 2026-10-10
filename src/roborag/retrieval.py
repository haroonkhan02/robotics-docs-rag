"""BM25, dense search, reciprocal-rank fusion and cross-encoder reranking behind one interface.

Modes (compared in the evaluation):
    bm25           lexical only: exact words, great for `ros2 topic echo`, `cmd_vel`, flags
    dense          embedding only: paraphrases and concepts ("make two nodes talk")
    hybrid         both lists merged with reciprocal-rank fusion (RRF)
    hybrid_rerank  hybrid candidates re-scored by a cross-encoder
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from roborag.chunking import Chunk

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")
TOKEN_RE = re.compile(r"[a-z0-9_]+")
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how",
    "i", "if", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was",
    "what", "when", "where", "which", "who", "why", "will", "with", "you", "your",
})


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens; `cmd_vel` is kept whole *and* split, so both forms match."""
    out: list[str] = []
    for tok in TOKEN_RE.findall(text.lower()):
        if tok in STOPWORDS:
            continue
        out.append(tok)
        if "_" in tok:
            out += [p for p in tok.split("_") if p and p not in STOPWORDS]
    return out


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self.ids = [c.chunk_id for c in chunks]
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks], k1=k1, b=b)

    def query(self, text: str, n: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(text))
        top = np.argsort(-scores)[:n]
        return [(self.ids[i], float(scores[i])) for i in top if scores[i] > 0]


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal-rank fusion: score(d) = sum over lists of 1 / (k + rank(d)), rank from 1.

    Uses ranks only, so BM25 scores (unbounded) and cosine similarities (-1..1) never have
    to be put on a common scale.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


@dataclass
class Hit:
    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # rank in each component retriever


@dataclass
class SearchResult:
    hits: list[Hit]
    timings_ms: dict[str, float]


class Retriever:
    def __init__(self, chunks: Sequence[Chunk], embedder=None, store=None, reranker=None,
                 candidates: int = 30, rrf_k: int = 60, rerank_top: int = 30):
        self.chunks = {c.chunk_id: c for c in chunks}
        self.bm25 = BM25Index(chunks)
        self.embedder, self.store, self.reranker = embedder, store, reranker
        self.candidates, self.rrf_k, self.rerank_top = candidates, rrf_k, rerank_top

    def _dense(self, query: str, n: int) -> list[tuple[str, float]]:
        if self.embedder is None or self.store is None:
            raise RuntimeError("dense retrieval needs an embedder and a vector store")
        q = self.embedder.encode_queries([query])[0]
        return self.store.query(q, n)

    def search(self, query: str, mode: str = "hybrid_rerank", k: int = 5) -> SearchResult:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
        t: dict[str, float] = {}
        n = max(self.candidates, k)
        ranked: list[tuple[str, float]]
        ranks: dict[str, dict[str, int]] = {}
        if mode in ("bm25", "hybrid", "hybrid_rerank"):
            t0 = time.perf_counter()
            bm = self.bm25.query(query, n)
            t["bm25"] = (time.perf_counter() - t0) * 1e3
            for r, (cid, _) in enumerate(bm, 1):
                ranks.setdefault(cid, {})["bm25"] = r
        if mode in ("dense", "hybrid", "hybrid_rerank"):
            t0 = time.perf_counter()
            de = self._dense(query, n)
            t["dense"] = (time.perf_counter() - t0) * 1e3
            for r, (cid, _) in enumerate(de, 1):
                ranks.setdefault(cid, {})["dense"] = r
        if mode == "bm25":
            ranked = bm
        elif mode == "dense":
            ranked = de
        else:
            ranked = rrf_fuse([[c for c, _ in bm], [c for c, _ in de]], self.rrf_k)
        if mode == "hybrid_rerank":
            if self.reranker is None:
                raise RuntimeError("hybrid_rerank needs a reranker")
            t0 = time.perf_counter()
            pool = [cid for cid, _ in ranked[: self.rerank_top]]
            scores = self.reranker.score(query, [self.chunks[c].text for c in pool])
            ranked = sorted(zip(pool, scores, strict=True), key=lambda kv: -kv[1])
            t["rerank"] = (time.perf_counter() - t0) * 1e3
        hits = [Hit(self.chunks[cid], s, ranks.get(cid, {})) for cid, s in ranked[:k]
                if cid in self.chunks]
        t["total"] = sum(t.values())
        return SearchResult(hits, t)
