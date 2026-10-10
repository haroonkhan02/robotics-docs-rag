import pytest

from roborag.chunking import chunk_corpus, chunk_document, split_recursive, split_words


def test_split_words_overlap():
    text = " ".join(str(i) for i in range(10))
    parts = split_words(text, 4, 1)
    assert parts == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]
    with pytest.raises(ValueError):
        split_words(text, 3, 3)


def test_recursive_respects_limit_and_paragraphs():
    paras = ["word " * 30, "other " * 30, "third " * 50]
    out = split_recursive("\n\n".join(p.strip() for p in paras), 70)
    assert all(len(c.split()) <= 70 for c in out)
    assert out[0].count("word") == 30 and "other" in out[0]  # 30 + 30 packed together
    assert out[1].startswith("third")


def test_long_sentence_falls_back_to_words():
    out = split_recursive("x " * 250, 100)
    assert [len(c.split()) for c in out] == [100, 100, 50]


def test_section_chunks_carry_heading_prefix(docs):
    chunks = chunk_document(docs[0], "section", max_words=60)
    assert chunks[1].text.startswith("Topics > Durability\n\n")
    assert chunks[1].section == "Topics > Durability"
    plain = chunk_document(docs[0], "section", max_words=60, section_header=False)
    assert plain[1].text.startswith("Transient local")


def test_pdf_chunks_never_span_pages(docs):
    for strategy in ("fixed", "recursive", "section"):
        chunks = chunk_document(docs[3], strategy, max_words=60, overlap_words=10)
        assert [c.page for c in chunks] == [1, 2]


def test_chunk_ids_unique_and_strategy_checked(docs):
    for strategy in ("fixed", "recursive", "section"):
        ids = [c.chunk_id for c in chunk_corpus(docs, strategy, 20, 5)]
        assert len(ids) == len(set(ids))
    with pytest.raises(ValueError):
        chunk_document(docs[0], "semantic")


def test_fixed_chunks_locate_their_section(docs):
    chunks = chunk_document(docs[1], "fixed", max_words=8, overlap_words=0)
    assert chunks[0].section == "Services"
    assert any(c.section == "Services > Call" for c in chunks)
