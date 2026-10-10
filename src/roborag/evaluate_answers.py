"""`rag-eval-answers`: end-to-end answer quality, judged by Claude against the passages.

For every question the pipeline answers with citations; a judge call (structured JSON output)
then splits the answer into atomic claims and checks each one against (a) the passages it
cites and (b) all passages shown to the model. It also grades the answer against the
reference answer and records whether the model abstained.

    faithfulness        claims supported by the retrieved passages / all claims
    citation accuracy   claims supported by the passages they cite / claims with a citation
    uncited claims      claims with no citation / all claims
    correctness         judge verdict vs. the reference (correct = 1, partial = 0.5)
    abstention          answerable: false-abstention rate; unanswerable: correct-abstention rate

Needs ANTHROPIC_API_KEY. Without it the command explains what to do and exits cleanly.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from roborag.config import load_config
from roborag.evalset import DEFAULT_EVALSET, Question, load_questions
from roborag.generate import Answer, format_passages, has_api_key

JUDGE_SYSTEM = """You are a strict evaluator of a retrieval-augmented question-answering system.
You receive a question, a reference answer written by a human, the numbered passages the system
was shown, and the system's answer, which cites passages as [n].

1. Split the answer into atomic factual claims (ignore filler such as "Here is how").
   For each claim list the passage numbers it cites, then decide:
   - supported_by_cited: every part of the claim is stated or directly implied by the cited
     passages (false if it cites nothing);
   - supported_by_context: every part of the claim is stated or directly implied by at least
     one of the passages, cited or not.
2. abstained: true if the answer says it cannot answer from the documentation.
3. correctness against the reference answer: "correct" (contains the key facts and nothing
   that contradicts them), "partially_correct", "incorrect", or "not_applicable" if the
   system abstained.
Judge only against the passages and the reference, never against your own knowledge."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "cited": {"type": "array", "items": {"type": "integer"}},
                    "supported_by_cited": {"type": "boolean"},
                    "supported_by_context": {"type": "boolean"},
                },
                "required": ["claim", "cited", "supported_by_cited", "supported_by_context"],
                "additionalProperties": False,
            },
        },
        "abstained": {"type": "boolean"},
        "correctness": {"type": "string",
                        "enum": ["correct", "partially_correct", "incorrect",
                                 "not_applicable"]},
        "explanation": {"type": "string"},
    },
    "required": ["claims", "abstained", "correctness", "explanation"],
    "additionalProperties": False,
}
CORRECTNESS_SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


def judge_prompt(q: Question, ans: Answer) -> str:
    return (f"<question>\n{q.question}\n</question>\n\n"
            f"<reference_answer>\n{q.reference_answer}\n</reference_answer>\n\n"
            f"<passages>\n{format_passages(ans.hits)}\n</passages>\n\n"
            f"<system_answer>\n{ans.text}\n</system_answer>")


def judge(client, model: str, q: Question, ans: Answer) -> dict:
    response = client.messages.create(
        model=model,
        max_tokens=8000,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": judge_prompt(q, ans)}],
        output_config={"effort": "medium",
                       "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
    )
    if response.stop_reason == "refusal":
        return {"claims": [], "abstained": False, "correctness": "not_applicable",
                "explanation": "judge declined", "judge_refused": True}
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def score_item(q: Question, verdict: dict) -> dict:
    claims = verdict.get("claims", [])
    n = len(claims)
    cited = [c for c in claims if c["cited"]]
    return {
        "faithfulness": (sum(c["supported_by_context"] for c in claims) / n) if n else None,
        "citation_accuracy": (sum(c["supported_by_cited"] for c in cited) / len(cited))
        if cited else None,
        "uncited_rate": (sum(not c["cited"] for c in claims) / n) if n else None,
        "correctness": CORRECTNESS_SCORE.get(verdict.get("correctness", ""))
        if q.answerable else None,
        "abstained": bool(verdict.get("abstained")),
        "n_claims": n,
    }


def aggregate(items: list[dict]) -> dict:
    def mean(key, subset):
        vals = [i[key] for i in subset if i[key] is not None]
        return round(statistics.mean(vals), 3) if vals else None

    ans = [i for i in items if i["answerable"]]
    una = [i for i in items if not i["answerable"]]
    return {
        "n_answerable": len(ans),
        "n_unanswerable": len(una),
        "faithfulness": mean("faithfulness", items),
        "citation_accuracy": mean("citation_accuracy", items),
        "uncited_claim_rate": mean("uncited_rate", items),
        "correctness": mean("correctness", ans),
        "false_abstention_rate": round(sum(i["abstained"] for i in ans) / len(ans), 3)
        if ans else None,
        "correct_abstention_rate": round(sum(i["abstained"] for i in una) / len(una), 3)
        if una else None,
        "invalid_citation_answers": sum(bool(i["invalid_citations"]) for i in items),
        "mean_answer_latency_ms": round(statistics.mean(i["latency_ms"] for i in items), 1)
        if items else None,
    }


def run(pipeline, client, judge_model: str, questions: list[Question]) -> dict:
    items = []
    for q in questions:
        ans = pipeline.ask(q.question)
        verdict = judge(client, judge_model, q, ans)
        s = score_item(q, verdict)
        items.append({"id": q.id, "category": q.category, "answerable": q.answerable,
                      "question": q.question, "answer": ans.text, "citations": ans.citations,
                      "invalid_citations": ans.invalid_citations,
                      "passages": [h.chunk.chunk_id for h in ans.hits],
                      "latency_ms": round(ans.latency_ms, 1), "usage": ans.usage,
                      "verdict": verdict, **s})
        print(f"{q.id}: faith={s['faithfulness']} correct={s['correctness']} "
              f"abstained={s['abstained']}")
    return {"summary": aggregate(items), "items": items}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--questions", default=str(DEFAULT_EVALSET))
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions")
    ap.add_argument("--judge-model", default=None, help="defaults to generation.model")
    ap.add_argument("--out", default="reports/answers/results.json")
    args = ap.parse_args(argv)
    if not has_api_key():
        print("ANTHROPIC_API_KEY is not set, so answer generation and the LLM judge are "
              "skipped.\nPut the key in your shell environment or in a local .env (see "
              ".env.example), then re-run `rag-eval-answers`. Retrieval metrics do not need "
              "a key: `rag-eval-retrieval`.")
        return
    from roborag.generate import make_client, pipeline_from_config

    cfg = load_config(args.config, args.set)
    questions = load_questions(args.questions)[: args.limit]
    pipeline = pipeline_from_config(cfg)
    judge_model = args.judge_model or cfg["generation"]["model"]
    result = run(pipeline, make_client(), judge_model, questions)
    result["config"] = {"generation": cfg["generation"], "retrieval": cfg["retrieval"],
                        "chunking": cfg["chunking"], "judge_model": judge_model}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
