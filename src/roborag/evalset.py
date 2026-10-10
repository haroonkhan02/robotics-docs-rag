"""The evaluation set: hand-written questions with gold documents and answer keys.

A retrieved chunk counts as *relevant* when it comes from a gold document **and** contains
at least one answer key (a short phrase that answers the question). This definition does not
depend on how the corpus was chunked, which is what makes chunking strategies comparable.

`rag-check-eval` verifies that every question is answerable from the corpus as parsed.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from roborag.chunking import Chunk

DEFAULT_EVALSET = Path("eval/questions.jsonl")
_WS = re.compile(r"\s+")


def norm(text: str) -> str:
    return _WS.sub(" ", text.replace("\\", "")).strip().lower()


@dataclass
class Question:
    id: str
    category: str
    question: str
    gold_docs: list[str] = field(default_factory=list)
    answer_keys: list[str] = field(default_factory=list)
    reference_answer: str = ""

    @property
    def answerable(self) -> bool:
        return bool(self.gold_docs)


def load_questions(path: Path | str = DEFAULT_EVALSET) -> list[Question]:
    with Path(path).open(encoding="utf-8") as f:
        return [Question(**json.loads(line)) for line in f if line.strip()]


def contains_key(text: str, keys: list[str]) -> bool:
    t = norm(text)
    return any(norm(k) in t for k in keys)


def is_relevant(chunk: Chunk, q: Question) -> bool:
    return chunk.doc_id in q.gold_docs and contains_key(chunk.text, q.answer_keys)


def check_questions(questions: list[Question], docs_text: dict[str, str]) -> list[str]:
    """Return a list of problems (empty = every answerable question is grounded)."""
    problems = []
    seen = set()
    for q in questions:
        if q.id in seen:
            problems.append(f"{q.id}: duplicate id")
        seen.add(q.id)
        if not q.answerable:
            continue
        missing = [d for d in q.gold_docs if d not in docs_text]
        if missing:
            problems.append(f"{q.id}: gold doc(s) not in corpus: {missing}")
        if not q.answer_keys:
            problems.append(f"{q.id}: no answer keys")
        elif not any(contains_key(docs_text.get(d, ""), q.answer_keys) for d in q.gold_docs):
            problems.append(f"{q.id}: no answer key found in any gold doc")
    return problems


def main(argv: list[str] | None = None) -> None:
    from roborag.config import load_config
    from roborag.index import corpus_from_config

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--questions", default=str(DEFAULT_EVALSET))
    args = ap.parse_args(argv)
    qs = load_questions(args.questions)
    docs = {d.doc_id: d.text for d in corpus_from_config(load_config(args.config))}
    problems = check_questions(qs, docs)
    n_ans = sum(q.answerable for q in qs)
    print(f"{len(qs)} questions ({n_ans} answerable, {len(qs) - n_ans} unanswerable)")
    for p in problems:
        print("  PROBLEM", p)
    if problems:
        raise SystemExit(1)
    print("all answerable questions are grounded in their gold documents")


if __name__ == "__main__":
    main()
