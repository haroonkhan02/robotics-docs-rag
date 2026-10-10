"""Grounded answer generation with Claude: numbered passages in, cited answer out.

The API key is read by the Anthropic SDK from `ANTHROPIC_API_KEY` (or a local, git-ignored
`.env` loaded here). It is never logged. Without a key the pipeline still runs and returns
the retrieved passages, so search, the UI and the retrieval evaluation work offline.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from roborag.metrics import parse_citations
from roborag.retrieval import Hit

ABSTAIN = "I can't answer that from the ROS 2 documentation I have."
SYSTEM_PROMPT = f"""You answer questions about ROS 2 (Robot Operating System 2) for robotics \
engineers, using only the numbered documentation passages supplied with each question.

Rules:
- Every sentence that states a fact must end with the number(s) of the passage(s) that \
support it, in square brackets, e.g. "Use transient local durability [2]." or "[1][3]".
- Use only information from the passages. Do not add facts from memory, even if you know them.
- If the passages do not contain the answer, reply with exactly: "{ABSTAIN}"
- Be concise: answer first, then any needed detail. Put commands, code and names in \
backticks. Copy commands exactly as they appear in the passages."""

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def has_api_key() -> bool:
    try:
        from dotenv import find_dotenv, load_dotenv

        # a .env in the working directory only fills variables that are not already set
        load_dotenv(find_dotenv(usecwd=True), override=False)
    except ImportError:  # pragma: no cover - python-dotenv is a dependency
        pass
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def format_passages(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        c = h.chunk
        where = c.section + (f" (page {c.page})" if c.page and "page" not in c.section else "")
        blocks.append(f"[{i}] {where}\nSource: {c.source}\n{c.text}")
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    content = (f"<passages>\n{format_passages(hits)}\n</passages>\n\n"
               f"Question: {question}")
    return [{"role": "user", "content": content}]


@dataclass
class Answer:
    question: str
    text: str
    citations: list[int]  # 1-based passage numbers, order of first use
    hits: list[Hit]
    model: str | None
    abstained: bool
    llm_used: bool
    stop_reason: str | None = None
    latency_ms: float = 0.0
    usage: dict = field(default_factory=dict)

    @property
    def invalid_citations(self) -> list[int]:
        return [n for n in self.citations if not 1 <= n <= len(self.hits)]

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.text,
            "citations": self.citations,
            "invalid_citations": self.invalid_citations,
            "abstained": self.abstained,
            "llm_used": self.llm_used,
            "model": self.model,
            "stop_reason": self.stop_reason,
            "latency_ms": round(self.latency_ms, 1),
            "usage": self.usage,
            "passages": [
                {"n": i, "doc_id": h.chunk.doc_id, "section": h.chunk.section,
                 "source": h.chunk.source, "page": h.chunk.page, "score": round(h.score, 4),
                 "text": h.chunk.text}
                for i, h in enumerate(self.hits, start=1)
            ],
        }


def make_client():
    import anthropic

    return anthropic.Anthropic(max_retries=3)


class Generator:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low",
                 max_tokens: int = 4000, client=None):
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.client = client if client is not None else (make_client() if has_api_key()
                                                         else None)

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def answer(self, question: str, hits: list[Hit]) -> Answer:
        if not self.enabled:
            return Answer(question, "LLM disabled (no ANTHROPIC_API_KEY): showing the retrieved "
                          "passages only.", [], hits, None, False, False)
        if not hits:
            return Answer(question, ABSTAIN, [], hits, self.model, True, False)
        t0 = time.perf_counter()
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=build_messages(question, hits),
            output_config={"effort": self.effort},
            # on a safety-classifier decline, the API retries on a fallback model in-call
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        latency = (time.perf_counter() - t0) * 1e3
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if response.stop_reason == "refusal":
            text = text or "The model declined to answer this question."
        usage = {k: getattr(response.usage, k, None) for k in ("input_tokens", "output_tokens")}
        return Answer(
            question=question, text=text, citations=parse_citations(text), hits=hits,
            model=getattr(response, "model", self.model),
            abstained=ABSTAIN.lower().rstrip(".") in text.lower(), llm_used=True,
            stop_reason=response.stop_reason, latency_ms=latency, usage=usage,
        )


class RAGPipeline:
    """Retriever + Generator: the object the CLI, API and answer evaluation share."""

    def __init__(self, retriever, generator: Generator, mode: str = "hybrid_rerank",
                 top_k: int = 5):
        self.retriever, self.generator, self.mode, self.top_k = retriever, generator, mode, top_k

    def ask(self, question: str, mode: str | None = None, top_k: int | None = None) -> Answer:
        res = self.retriever.search(question, mode or self.mode, top_k or self.top_k)
        ans = self.generator.answer(question, res.hits)
        ans.usage["retrieval_ms"] = round(res.timings_ms["total"], 1)
        return ans


def pipeline_from_config(cfg: dict, generator: Generator | None = None) -> RAGPipeline:
    from roborag.index import open_retriever

    g, r = cfg["generation"], cfg["retrieval"]
    retriever = open_retriever(cfg)
    gen = generator or Generator(g["model"], g.get("effort", "low"), g.get("max_tokens", 4000))
    return RAGPipeline(retriever, gen, r["mode"], r["top_k"])
