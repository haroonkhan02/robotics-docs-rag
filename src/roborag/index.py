"""`rag-index`: corpus -> chunks -> embeddings -> Chroma, for one or all chunking strategies.

Also the single place that knows how to *open* an index (`open_retriever`), used by the
CLI, the API and the evaluation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from roborag.chunking import STRATEGIES, chunk_corpus
from roborag.config import load_config
from roborag.download import PAPER_TITLES
from roborag.ingest import load_corpus
from roborag.retrieval import Retriever
from roborag.store import ChromaStore, collection_name, load_chunks, save_chunks


def corpus_from_config(cfg: dict):
    c = cfg["corpus"]
    return load_corpus(Path(cfg["data_dir"]) / "ros2_documentation" / "source",
                       Path(c["pdf_dir"]), c.get("exclude_globs", []), PAPER_TITLES)


def chunks_path(cfg: dict, strategy: str) -> Path:
    return Path(cfg["index_dir"]) / f"chunks_{strategy}.jsonl"


def build_index(cfg: dict, strategy: str, embedder=None, docs=None) -> dict:
    from roborag.models import Embedder

    ch, emb_cfg = cfg["chunking"], cfg["embedding"]
    docs = docs if docs is not None else corpus_from_config(cfg)
    t0 = time.perf_counter()
    chunks = chunk_corpus(docs, strategy, ch["max_words"], ch["overlap_words"],
                          ch.get("section_header", True))
    t_chunk = time.perf_counter() - t0
    embedder = embedder or Embedder(emb_cfg["model"], emb_cfg.get("query_prefix", ""),
                                    emb_cfg.get("batch_size", 64))
    t0 = time.perf_counter()
    vectors = embedder.encode_docs([c.text for c in chunks], progress=True)
    t_embed = time.perf_counter() - t0
    store = ChromaStore(Path(cfg["index_dir"]) / "chroma",
                        collection_name(strategy, emb_cfg["model"]))
    store.reset()
    t0 = time.perf_counter()
    store.add(chunks, vectors)
    t_store = time.perf_counter() - t0
    save_chunks(chunks, chunks_path(cfg, strategy))
    lengths = sorted(len(c.text.split()) for c in chunks)
    stats = {
        "strategy": strategy,
        "embedding_model": emb_cfg["model"],
        "documents": len(docs),
        "chunks": len(chunks),
        "words_median": lengths[len(lengths) // 2],
        "words_p95": lengths[int(len(lengths) * 0.95)],
        "words_max": lengths[-1],
        "seconds_chunk": round(t_chunk, 2),
        "seconds_embed": round(t_embed, 1),
        "seconds_store": round(t_store, 1),
    }
    print(json.dumps(stats))
    return stats


def open_retriever(cfg: dict, strategy: str | None = None, embedder=None, reranker=None,
                   need_reranker: bool | None = None) -> Retriever:
    from roborag.models import Embedder, Reranker

    strategy = strategy or cfg["chunking"]["strategy"]
    emb_cfg, r = cfg["embedding"], cfg["retrieval"]
    path = chunks_path(cfg, strategy)
    if not path.exists():
        raise SystemExit(f"no index for strategy {strategy!r}: run `rag-index --strategy "
                         f"{strategy}` first")
    chunks = load_chunks(path)
    embedder = embedder or Embedder(emb_cfg["model"], emb_cfg.get("query_prefix", ""))
    store = ChromaStore(Path(cfg["index_dir"]) / "chroma",
                        collection_name(strategy, emb_cfg["model"]))
    if need_reranker is None:
        need_reranker = r["mode"] == "hybrid_rerank"
    if reranker is None and need_reranker:
        reranker = Reranker(r["reranker"])
    return Retriever(chunks, embedder, store, reranker, r["candidates"], r["rrf_k"],
                     r["rerank_top"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--strategy", default=None, help=f"one of {STRATEGIES} or 'all'")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. embedding.model=...")
    ap.add_argument("--stats-out", default=None, help="append index stats to this JSON file")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    strategies = STRATEGIES if args.strategy == "all" else [args.strategy
                                                            or cfg["chunking"]["strategy"]]
    from roborag.models import Embedder

    emb_cfg = cfg["embedding"]
    embedder = Embedder(emb_cfg["model"], emb_cfg.get("query_prefix", ""),
                        emb_cfg.get("batch_size", 64))
    docs = corpus_from_config(cfg)
    print(f"{len(docs)} documents, {sum(len(d.text.split()) for d in docs):,} words")
    all_stats = [build_index(cfg, s, embedder, docs) for s in strategies]
    if args.stats_out:
        out = Path(args.stats_out)
        old = json.loads(out.read_text()) if out.exists() else []
        keep = [o for o in old if (o["strategy"], o["embedding_model"]) not in
                {(s["strategy"], s["embedding_model"]) for s in all_stats}]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(keep + all_stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
