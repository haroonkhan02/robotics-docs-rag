"""`python -m roborag.ablations`: one table + figure from all `results_<tag>.json` files.

Rows: the section-chunking variants (main run, MiniLM embeddings, 100/300-word chunks, no
heading prefix) for every retrieval mode, plus the rank-of-first-relevant histogram.
"""

from __future__ import annotations

import json
from pathlib import Path

from roborag.metrics import mrr_at_k, paired_bootstrap_p
from roborag.plots import MODE_COLORS, plot_rank_histogram

VARIANTS = [  # (tag, label)
    ("main", "BGE-small, 180 words, heading prefix (default)"),
    ("emb_minilm", "MiniLM-L6 embeddings instead of BGE-small"),
    ("section_w100", "100-word chunks"),
    ("section_w300", "300-word chunks"),
    ("section_noheader", "no heading prefix"),
]


def load(dir_: Path, tag: str) -> dict | None:
    p = dir_ / f"results_{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main(dir_: str = "reports/retrieval") -> None:
    import matplotlib.pyplot as plt

    d = Path(dir_)
    rows, series = [], {}
    base = {}  # mode -> per-question MRR of the default run, for paired tests
    for tag, label in VARIANTS:
        data = load(d, tag)
        if data is None:
            continue
        for r in data["results"]:
            if r["strategy"] != "section":
                continue
            mrrs = [mrr_at_k(p["rels"], 10) for p in r["per_question"]]
            if tag == "main":
                base[r["mode"]] = mrrs
            p_val = (paired_bootstrap_p(base[r["mode"]], mrrs)
                     if tag != "main" and r["mode"] in base else None)
            rows.append((label, r["mode"], r["hit@5"]["mean"], r["mrr@10"]["mean"],
                         r["mrr@10"]["ci_low"], r["mrr@10"]["ci_high"], p_val,
                         r["latency_ms_mean"]))
            series.setdefault(r["mode"], []).append((label, r["mrr@10"]["mean"]))
    lines = ["| Variant (section chunking) | Retrieval | Hit@5 | MRR@10 (95% CI) "
             "| p vs default | ms/query |", "|---|---|---|---|---|---|"]
    for label, mode, h5, mrr, lo, hi, p_val, ms in rows:
        p_txt = "—" if p_val is None else ("< 0.001" if p_val < 0.001 else f"{p_val:.3f}")
        lines.append(f"| {label} | {mode} | {h5:.3f} | {mrr:.3f} ({lo:.2f}–{hi:.2f}) "
                     f"| {p_txt} | {ms:.0f} |")
    (d / "ablations.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    labels = [lbl for tag, lbl in VARIANTS if load(d, tag)]
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.8 / max(1, len(series))
    for j, (mode, vals) in enumerate(series.items()):
        xs = [labels.index(lbl) + (j - (len(series) - 1) / 2) * width for lbl, _ in vals]
        ax.bar(xs, [v for _, v in vals], width, label=mode, color=MODE_COLORS.get(mode))
    ticks = [lbl.replace(", ", ",\n").replace(" instead of", "\ninstead of") for lbl in labels]
    ax.set_xticks(range(len(labels)), ticks, fontsize=8)
    ax.set_ylabel("MRR@10")
    ax.set_ylim(0, 1)
    ax.set_title("Ablations on section chunking (61 questions; 95% CIs are about ±0.10)")
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "ablations.png", dpi=130)
    plt.close(fig)

    main_run = load(d, "main")
    if main_run:
        ranks = {f"{r['strategy']}/{r['mode']}": [p["first_relevant_rank"]
                                                  for p in r["per_question"]]
                 for r in main_run["results"]
                 if (r["strategy"], r["mode"]) in {("fixed", "bm25"), ("section", "dense"),
                                                   ("section", "hybrid_rerank")}}
        plot_rank_histogram(ranks, d / "first_relevant_rank.png")


if __name__ == "__main__":
    main()
