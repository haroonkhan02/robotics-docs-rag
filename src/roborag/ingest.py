"""Turn the raw corpus (reStructuredText docs and PDFs) into clean `Document` objects.

Every document keeps its section structure, because the section-aware chunker and the
citations shown to the user both need to know *where* in a document a passage came from.
"""

from __future__ import annotations

import fnmatch
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

DOCS_URL = "https://docs.ros.org/en/jazzy/{path}.html"

# Directives whose whole block (directive line + indented body) carries no answerable text.
SKIP_DIRECTIVES = {
    "contents", "image", "figure", "toctree", "redirect-from", "raw", "only", "include",
    "meta", "spelling", "spelling:word-list", "literalinclude", "youtube", "graphviz",
}
DIRECTIVE_RE = re.compile(r"^(\s*)\.\.\s+([\w:-]+)::\s*(.*)$")
OPTION_RE = re.compile(r"^\s+:[\w-]+:.*$")
ADORNMENT_RE = re.compile(r"^([=\-~^\"'`#*+<>_:.])\1{2,}\s*$")

INLINE_SUBS = [
    (re.compile(r":\w+:`([^`<]+?)\s*<[^`>]+>`"), r"\1"),  # :doc:`text <target>` -> text
    (re.compile(r":\w+:`~?([^`]+)`"), r"\1"),  # :ref:`target` / :class:`~a.B` -> target
    (re.compile(r"`([^`<]+?)\s*<[^`>]+>`__?"), r"\1"),  # `text <url>`_ -> text
    (re.compile(r"``([^`]+)``"), r"\1"),  # ``literal`` -> literal
    (re.compile(r"`([^`]+)`_{0,2}"), r"\1"),  # `interpreted`_ -> interpreted
    (re.compile(r"\*\*([^*]+)\*\*"), r"\1"),  # **bold**
    (re.compile(r"(?<![\w*])\*([^*\s][^*]*)\*(?![\w*])"), r"\1"),  # *italic*
    (re.compile(r"\|(\w[\w -]*)\|"), r"\1"),  # |substitution|
]


@dataclass
class Section:
    """A run of text under one heading path, e.g. ("Topics", "Publish/Subscribe")."""

    heading_path: tuple[str, ...]
    text: str
    page: int | None = None  # 1-based page for PDFs


@dataclass
class Document:
    doc_id: str  # stable id: path relative to the corpus root, e.g. "Concepts/Basic/About-Topics"
    title: str
    source: str  # URL (docs) or file path (PDF)
    kind: str  # "rst" | "pdf"
    sections: list[Section] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(s.text for s in self.sections if s.text)


def clean_inline(line: str) -> str:
    for pattern, repl in INLINE_SUBS:
        line = pattern.sub(repl, line)
    return line


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def strip_directives(lines: list[str]) -> list[str]:
    """Drop non-text directive blocks and directive/option lines; keep the body of the rest.

    `.. code-block:: console` keeps its commands, `.. note::` keeps its sentence, while
    `.. image::` and `.. toctree::` disappear with all of their indented options.
    """
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = DIRECTIVE_RE.match(line)
        if m:
            indent, name, arg = len(m.group(1)), m.group(2).lower(), m.group(3).strip()
            i += 1
            if name in SKIP_DIRECTIVES:
                while i < len(lines) and (not lines[i].strip() or _indent(lines[i]) > indent):
                    i += 1
                continue
            while i < len(lines) and OPTION_RE.match(lines[i]):
                i += 1
            if arg and name not in {"code-block", "code", "sourcecode", "tabs", "group-tab"}:
                out.append(" " * (indent + 3) + arg)  # e.g. ".. note:: Remember to ..."
            continue
        stripped = line.strip()
        if stripped.startswith(".. _") and stripped.endswith(":"):  # .. _label:
            i += 1
            continue
        if stripped == ".." or (stripped.startswith(".. ") and "::" not in stripped):
            # RST comment: skip it and its indented continuation
            i += 1
            while i < len(lines) and lines[i].strip() and _indent(lines[i]) > _indent(line):
                i += 1
            continue
        out.append(line)
        i += 1
    return out


def parse_rst(text: str) -> tuple[str, list[tuple[int, str, list[str]]]]:
    """Split RST into (title, [(level, heading, body_lines)]).

    Heading levels follow RST's rule: the order in which adornment styles first appear.
    """
    lines = strip_directives(text.splitlines())
    styles: list[str] = []
    blocks: list[tuple[int, str, list[str]]] = [(0, "", [])]
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        over = False
        if ADORNMENT_RE.match(line) and i + 2 < len(lines) and ADORNMENT_RE.match(lines[i + 2]):
            # over- and underlined title
            title_line, under, over = lines[i + 1], lines[i + 2], True
            if title_line.strip() and len(under.strip()) >= len(title_line.strip()):
                style = "o" + under.strip()[0]
                if style not in styles:
                    styles.append(style)
                blocks.append((styles.index(style) + 1, clean_inline(title_line.strip()), []))
                i += 3
                continue
        if (
            not over
            and line.strip()
            and not line.startswith(" ")
            and ADORNMENT_RE.match(nxt)
            and len(nxt.strip()) >= len(line.strip())
        ):
            style = nxt.strip()[0]
            if style not in styles:
                styles.append(style)
            blocks.append((styles.index(style) + 1, clean_inline(line.strip()), []))
            i += 2
            continue
        blocks[-1][2].append(line)
        i += 1
    title = next((h for lvl, h, _ in blocks if lvl == 1), blocks[1][1] if len(blocks) > 1 else "")
    return title, blocks


def _normalise_body(lines: list[str]) -> str:
    """Clean inline markup, drop indentation, collapse runs of blank lines."""
    paras: list[str] = []
    cur: list[str] = []
    for raw in lines:
        s = clean_inline(raw.strip())
        if not s:
            if cur:
                paras.append("\n".join(cur))
                cur = []
            continue
        if ADORNMENT_RE.match(s) or set(s) <= {"=", " ", "+", "-", "|"}:  # table borders
            continue
        cur.append(s)
    if cur:
        paras.append("\n".join(cur))
    return "\n\n".join(paras)


def load_rst(path: Path, root: Path) -> Document:
    rel = path.relative_to(root).with_suffix("").as_posix()
    title, blocks = parse_rst(path.read_text(encoding="utf-8", errors="replace"))
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    for level, heading, body in blocks:
        if level:
            stack = [(lvl, h) for lvl, h in stack if lvl < level] + [(level, heading)]
        text = _normalise_body(body)
        if text:
            path_ = tuple(h for _, h in stack) or (title or rel,)
            sections.append(Section(heading_path=path_, text=text))
    return Document(rel, title or rel, DOCS_URL.format(path=rel), "rst", sections)


def load_pdf(path: Path, title: str | None = None) -> Document:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    meta_title = (reader.metadata.title if reader.metadata else None) or ""
    title = title or meta_title.strip() or path.stem
    sections = []
    for n, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = unicodedata.normalize("NFKC", text)  # "ﬁ" ligature -> "fi"
        text = re.sub(r"-\n(\w)", r"\1", text)  # re-join hyphenated line breaks
        text = re.sub(r"[ \t]+", " ", text).strip()
        if text:
            sections.append(Section(heading_path=(title, f"page {n}"), text=text, page=n))
    return Document(f"pdf/{path.stem}", title, str(path), "pdf", sections)


def iter_rst_files(root: Path, exclude_globs: Iterable[str] = ()) -> list[Path]:
    files = []
    for p in sorted(root.rglob("*.rst")):
        rel = p.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel, g) for g in exclude_globs):
            continue
        files.append(p)
    return files


def load_corpus(
    rst_root: Path | None,
    pdf_dir: Path | None = None,
    exclude_globs: Iterable[str] = (),
    pdf_titles: dict[str, str] | None = None,
) -> list[Document]:
    docs: list[Document] = []
    if rst_root and rst_root.exists():
        docs += [load_rst(p, rst_root) for p in iter_rst_files(rst_root, exclude_globs)]
    if pdf_dir and pdf_dir.exists():
        for p in sorted(pdf_dir.glob("*.pdf")):
            docs.append(load_pdf(p, (pdf_titles or {}).get(p.stem)))
    return [d for d in docs if d.text.strip()]
