"""Three chunking strategies, compared in the retrieval evaluation.

* ``fixed``     – sliding windows of N words with overlap over the whole document.
                  The baseline every RAG tutorial starts with; ignores all structure.
* ``recursive`` – greedy packing of paragraphs, falling back to sentences, then words,
                  so that chunks end at natural boundaries (no overlap).
* ``section``   – recursive packing *inside* each heading section, and every chunk is
                  prefixed with its heading path ("Doc title > Section > Subsection"), so a
                  chunk that only says "Run the command below" still carries its topic.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

from roborag.ingest import Document

STRATEGIES = ("fixed", "recursive", "section")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9$`(\"'])")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str  # "Title > Section > Sub"
    text: str  # what gets embedded, indexed by BM25 and shown to the LLM
    source: str
    page: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _words(text: str) -> list[str]:
    return text.split()


def _chunk_id(doc_id: str, idx: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{doc_id}#{idx}-{digest}"


def split_words(text: str, size: int, overlap: int = 0) -> list[str]:
    """Windows of `size` words, each starting `size - overlap` words after the last."""
    if size <= 0 or overlap >= size:
        raise ValueError("need size > 0 and overlap < size")
    words = _words(text)
    if not words:
        return []
    step = size - overlap
    out = []
    for start in range(0, len(words), step):
        out.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return out


def split_recursive(text: str, max_words: int) -> list[str]:
    """Pack paragraphs up to `max_words`; split oversize paragraphs by sentence, then by words.

    Newlines inside a paragraph are kept, so code blocks stay readable for the LLM.
    """
    pieces: list[str] = []
    for para in (p.strip() for p in text.split("\n\n")):
        if not para:
            continue
        if len(_words(para)) <= max_words:
            pieces.append(para)
            continue
        for sent in SENTENCE_RE.split(para):
            if len(_words(sent)) <= max_words:
                pieces.append(sent)
            else:
                pieces.extend(split_words(sent, max_words))
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for piece in pieces:
        n = len(_words(piece))
        if cur and cur_len + n > max_words:
            chunks.append("\n\n".join(cur))
            cur, cur_len = [], 0
        cur.append(piece)
        cur_len += n
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def _make(doc: Document, idx: int, text: str, section: str, page: int | None) -> Chunk:
    return Chunk(_chunk_id(doc.doc_id, idx, text), doc.doc_id, doc.title, section, text,
                 doc.source, page)


def _locate(doc: Document, text: str) -> tuple[str, int | None]:
    """Heading path and page of the section where `text` starts (for fixed/recursive)."""
    head = " ".join(_words(text)[:12])
    for s in doc.sections:
        if head and head in " ".join(_words(s.text)):
            return " > ".join(s.heading_path), s.page
    first = doc.sections[0] if doc.sections else None
    return (" > ".join(first.heading_path), first.page) if first else (doc.title, None)


def chunk_document(
    doc: Document, strategy: str, max_words: int = 180, overlap_words: int = 40,
    section_header: bool = True,
) -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
    chunks: list[Chunk] = []
    if strategy == "section":
        for s in doc.sections:
            header = " > ".join(s.heading_path)
            # leave room for the header so the chunk stays within max_words overall
            body_budget = (max(40, max_words - len(_words(header))) if section_header
                           else max_words)
            for body in split_recursive(s.text, body_budget):
                text = f"{header}\n\n{body}" if section_header else body
                chunks.append(_make(doc, len(chunks), text, header, s.page))
        return chunks
    if doc.kind == "pdf":
        # never let a chunk span two pages, so page citations stay exact
        texts = []
        for s in doc.sections:
            parts = (split_words(s.text, max_words, overlap_words) if strategy == "fixed"
                     else split_recursive(s.text, max_words))
            texts += [(t, " > ".join(s.heading_path), s.page) for t in parts]
        return [_make(doc, i, t, sec, pg) for i, (t, sec, pg) in enumerate(texts)]
    parts = (split_words(doc.text, max_words, overlap_words) if strategy == "fixed"
             else split_recursive(doc.text, max_words))
    for t in parts:
        section, page = _locate(doc, t)
        chunks.append(_make(doc, len(chunks), t, section, page))
    return chunks


def chunk_corpus(docs: list[Document], strategy: str, max_words: int = 180,
                 overlap_words: int = 40, section_header: bool = True) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out += chunk_document(d, strategy, max_words, overlap_words, section_header)
    return out
