import pytest

from roborag.retrieval import BM25Index, Retriever, rrf_fuse, tokenize
from roborag.store import ChromaStore, load_chunks, save_chunks
from tests.conftest import LengthReranker


def test_tokenize_keeps_and_splits_snake_case():
    toks = tokenize("How do I echo /turtle1/cmd_vel?")
    assert "cmd_vel" in toks and "cmd" in toks and "vel" in toks
    assert "how" not in toks and "do" not in toks  # stopwords


def test_rrf_fuse_rewards_agreement():
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    order = [cid for cid, _ in fused]
    assert set(order[:2]) == {"a", "b"} and order[-1] in {"c", "d"}
    score = dict(fused)
    assert score["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert score["c"] == pytest.approx(1 / 63)


def test_bm25_finds_exact_terms(chunks):
    bm = BM25Index(chunks)
    top_id, _ = bm.query("ros2 service call", 3)[0]
    assert "Services" in top_id
    assert bm.query("zzzz unknown", 3) == []


@pytest.fixture
def retriever(tmp_path, chunks, embedder):
    store = ChromaStore(tmp_path / "chroma", "test-hash")
    store.reset()
    store.add(chunks, embedder.encode_docs([c.text for c in chunks]))
    assert store.count() == len(chunks)
    return Retriever(chunks, embedder, store, LengthReranker(), candidates=10, rerank_top=10)


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "hybrid_rerank"])
def test_every_mode_returns_ranked_hits(retriever, mode):
    res = retriever.search("transient local late-joining subscriptions", mode, k=3)
    assert 1 <= len(res.hits) <= 3
    assert res.timings_ms["total"] >= 0
    if mode != "hybrid_rerank":
        assert res.hits[0].chunk.section == "Topics > Durability"


def test_rerank_reorders_by_reranker_score(retriever):
    hits = retriever.search("topics services bags", "hybrid_rerank", k=5).hits
    lengths = [len(h.chunk.text) for h in hits]
    assert lengths == sorted(lengths)  # LengthReranker prefers short passages
    assert all("bm25" in h.ranks or "dense" in h.ranks for h in hits)


def test_dense_scores_are_cosine(retriever, embedder):
    query = "Record topics with ros2 bag record -o name"
    hits = retriever.search(query, "dense", k=1).hits
    assert hits[0].chunk.doc_id == "Tutorials/Bags"
    q, d = embedder.encode_queries([query])[0], embedder.encode_docs([hits[0].chunk.text])[0]
    assert hits[0].score == pytest.approx(float(q @ d), abs=1e-4)


def test_unknown_mode_and_missing_reranker(chunks, embedder):
    r = Retriever(chunks)
    with pytest.raises(ValueError):
        r.search("x", "magic")
    with pytest.raises(RuntimeError):
        r.search("x", "dense")


def test_chunks_roundtrip(tmp_path, chunks):
    save_chunks(chunks, tmp_path / "c.jsonl")
    assert load_chunks(tmp_path / "c.jsonl") == chunks
