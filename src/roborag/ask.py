"""`rag-ask "question"`: retrieve, answer with citations, print sources."""

from __future__ import annotations

import argparse
import json
import textwrap

from roborag.config import load_config
from roborag.retrieval import MODES


def main(argv: list[str] | None = None) -> None:
    from roborag.generate import pipeline_from_config

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--mode", choices=MODES, default=None)
    ap.add_argument("-k", "--top-k", type=int, default=None)
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = ap.parse_args(argv)
    pipe = pipeline_from_config(load_config(args.config, args.set))
    ans = pipe.ask(args.question, args.mode, args.top_k)
    if args.json:
        print(json.dumps(ans.to_dict(), indent=2, ensure_ascii=False))
        return
    print(textwrap.fill(ans.text, 100, replace_whitespace=False) if ans.llm_used else ans.text)
    print("\nSources:")
    cited = set(ans.citations)
    for i, h in enumerate(ans.hits, start=1):
        mark = "*" if i in cited else " "
        page = f" p.{h.chunk.page}" if h.chunk.page and "page" not in h.chunk.section else ""
        print(f" {mark}[{i}] {h.chunk.section}{page}\n      {h.chunk.source}")
    if ans.invalid_citations:
        print(f"\nwarning: answer cites passages that do not exist: {ans.invalid_citations}")


if __name__ == "__main__":
    main()
