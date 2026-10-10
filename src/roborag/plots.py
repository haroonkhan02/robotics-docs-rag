"""Figures for the README and reports (matplotlib, Agg backend, no display needed)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODE_COLORS = {"bm25": "#8c8c8c", "dense": "#4c72b0", "hybrid": "#55a868",
               "hybrid_rerank": "#c44e52"}


def plot_retrieval(results: list[dict], path: Path) -> None:
    strategies = list(dict.fromkeys(r["strategy"] for r in results))
    modes = list(dict.fromkeys(r["mode"] for r in results))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    width = 0.8 / len(modes)
    for ax, metric, label in ((axes[0], "hit@5", "Hit@5"), (axes[1], "mrr@10", "MRR@10")):
        for j, mode in enumerate(modes):
            xs, ys, lo, hi = [], [], [], []
            for i, s in enumerate(strategies):
                r = next((r for r in results if r["strategy"] == s and r["mode"] == mode), None)
                if r is None:
                    continue
                m = r[metric]
                xs.append(i + (j - (len(modes) - 1) / 2) * width)
                ys.append(m["mean"])
                lo.append(m["mean"] - m["ci_low"])
                hi.append(m["ci_high"] - m["mean"])
            ax.bar(xs, ys, width, yerr=[lo, hi], capsize=2, label=mode,
                   color=MODE_COLORS.get(mode), edgecolor="white")
        ax.set_xticks(range(len(strategies)), strategies)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(label)
        ax.set_title(f"{label} by chunking strategy (95% bootstrap CI)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper center", ncol=4)
    ax = axes[2]
    for r in results:
        ax.scatter(r["latency_ms_mean"], r["mrr@10"]["mean"], color=MODE_COLORS.get(r["mode"]),
                   s=40)
        ax.annotate(r["strategy"][:3], (r["latency_ms_mean"], r["mrr@10"]["mean"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("mean latency per query (ms, log scale)")
    ax.set_ylabel("MRR@10")
    ax.set_title("Quality vs. latency")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_rank_histogram(per_question: dict[str, list[int | None]], path: Path,
                        k: int = 10) -> None:
    """Where the first relevant chunk lands, per configuration (None = not in top k)."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    names = list(per_question)
    width = 0.8 / len(names)
    for j, name in enumerate(names):
        ranks = per_question[name]
        counts = [sum(1 for r in ranks if r == i) for i in range(1, k + 1)]
        counts.append(sum(1 for r in ranks if r is None))
        xs = [i + (j - (len(names) - 1) / 2) * width for i in range(k + 1)]
        ax.bar(xs, counts, width, label=name, color=MODE_COLORS.get(name.split("/")[-1]))
    ax.set_xticks(range(k + 1), [str(i) for i in range(1, k + 1)] + ["miss"])
    ax.set_xlabel("rank of the first relevant chunk")
    ax.set_ylabel("questions")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
