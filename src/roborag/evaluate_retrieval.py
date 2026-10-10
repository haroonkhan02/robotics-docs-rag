"""`rag-eval-retrieval`: every chunking strategy x every retrieval mode on the eval set.

Writes `reports/retrieval/results_<tag>.json` (aggregate + per-question ranks), a markdown
table and plots. Needs the indexes from `rag-index --strategy all`; no API key, no LLM.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from roborag.chunking import STRATEGIES
from roborag.config import load_config
from roborag.evalset import DEFAULT_EVALSET, is_relevant, load_questions
from roborag.metrics import (
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    paired_bootstrap_p,
    precision_at_k,
    summarize,
)
from roborag.retrieval import MODES

KS = (1, 3, 5, 10)


def evaluate_config(retriever, questions, mode: str, k: int = 10) -> dict:
    chunks = list(retriever.chunks.values())
    n_rel = {q.id: sum(is_relevant(c, q) for c in chunks) for q in questions}
    retriever.search(questions[0].question, mode, k)  # warm-up (model load, MPS kernels)
    per_q = []
    for q in questions:
        res = retriever.search(q.question, mode, k)
        rels = [is_relevant(h.chunk, q) for h in res.hits]
        rels += [False] * (k - len(rels))
        first = next((i for i, r in enumerate(rels, 1) if r), None)
        per_q.append({
            "id": q.id, "category": q.category, "rels": rels, "first_relevant_rank": first,
            "n_relevant_in_index": n_rel[q.id],
            "top_docs": [h.chunk.doc_id for h in res.hits[:5]],
            "latency_ms": round(res.timings_ms["total"], 2),
        })
    return {"mode": mode, "per_question": per_q, **aggregate(per_q, k)}


def aggregate(per_q: list[dict], k: int = 10) -> dict:
    out: dict = {}
    for kk in KS:
        out[f"hit@{kk}"] = summarize([hit_at_k(p["rels"], kk) for p in per_q])
    out[f"mrr@{k}"] = summarize([mrr_at_k(p["rels"], k) for p in per_q])
    out[f"ndcg@{k}"] = summarize([ndcg_at_k(p["rels"], k, p["n_relevant_in_index"])
                                  for p in per_q])
    out["p@5"] = summarize([precision_at_k(p["rels"], 5) for p in per_q])
    lat = sorted(p["latency_ms"] for p in per_q)
    out["latency_ms_mean"] = round(statistics.mean(lat), 1)
    out["latency_ms_p95"] = round(lat[int(0.95 * (len(lat) - 1))], 1)
    cats = sorted({p["category"] for p in per_q})
    out["hit@5_by_category"] = {
        c: round(statistics.mean(hit_at_k(p["rels"], 5) for p in per_q if p["category"] == c), 3)
        for c in cats
    }
    out["n_questions"] = len(per_q)
    return out


def significance(results: list[dict]) -> list[dict]:
    """Paired bootstrap of every config against the best one on per-question MRR@10."""
    def scores(r):
        return [mrr_at_k(p["rels"], 10) for p in r["per_question"]]

    best = max(results, key=lambda r: r["mrr@10"]["mean"])
    rows = []
    for r in results:
        if r is best:
            continue
        rows.append({"best": f"{best['strategy']}/{best['mode']}",
                     "other": f"{r['strategy']}/{r['mode']}",
                     "mrr_diff": round(best["mrr@10"]["mean"] - r["mrr@10"]["mean"], 3),
                     "p_value": paired_bootstrap_p(scores(best), scores(r))})
    return rows


def markdown_table(results: list[dict]) -> str:
    head = ("| Chunking | Retrieval | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 | nDCG@10 | P@5 "
            "| ms/query |\n|---|---|---|---|---|---|---|---|---|---|")
    lines = [head]
    for r in results:
        f = [r[f"hit@{k}"]["mean"] for k in KS] + [r["mrr@10"]["mean"], r["ndcg@10"]["mean"],
                                                   r["p@5"]["mean"]]
        lines.append(f"| {r['strategy']} | {r['mode']} | " + " | ".join(f"{v:.3f}" for v in f)
                     + f" | {r['latency_ms_mean']:.0f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    from roborag.index import open_retriever
    from roborag.models import Embedder, Reranker

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=STRATEGIES)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--questions", default=str(DEFAULT_EVALSET))
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out", default="reports/retrieval")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    questions = [q for q in load_questions(args.questions) if q.answerable]
    emb_cfg, r_cfg = cfg["embedding"], cfg["retrieval"]
    embedder = Embedder(emb_cfg["model"], emb_cfg.get("query_prefix", ""))
    reranker = Reranker(r_cfg["reranker"]) if "hybrid_rerank" in args.modes else None
    results = []
    for strategy in args.strategies:
        retriever = open_retriever(cfg, strategy, embedder, reranker, need_reranker=False)
        for mode in args.modes:
            res = evaluate_config(retriever, questions, mode)
            res["strategy"] = strategy
            results.append(res)
            print(f"{strategy:9s} {mode:13s} hit@5={res['hit@5']['mean']:.3f} "
                  f"mrr@10={res['mrr@10']['mean']:.3f} ndcg@10={res['ndcg@10']['mean']:.3f} "
                  f"{res['latency_ms_mean']:.0f} ms")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag": args.tag,
        "embedding_model": emb_cfg["model"],
        "reranker": r_cfg["reranker"],
        "chunking": cfg["chunking"],
        "retrieval": {k: r_cfg[k] for k in ("candidates", "rrf_k", "rerank_top")},
        "n_questions": len(questions),
        "significance_vs_best": significance(results) if len(results) > 1 else [],
        "results": results,
    }
    (out / f"results_{args.tag}.json").write_text(json.dumps(payload, indent=1) + "\n")
    (out / f"table_{args.tag}.md").write_text(markdown_table(results))
    print(markdown_table(results))
    if len(args.strategies) > 1:
        from roborag.plots import plot_retrieval

        plot_retrieval(results, out / f"retrieval_{args.tag}.png")


if __name__ == "__main__":
    main()
