"""FastAPI backend + a one-page chat UI. `rag-serve` starts it on http://127.0.0.1:8000.

    GET  /            chat UI (static/index.html)
    GET  /health      index + model status (never reveals the API key)
    POST /search      {"query", "mode"?, "k"?}       -> ranked passages
    POST /ask         {"question", "mode"?, "k"?}    -> answer + citations + passages
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from roborag.retrieval import MODES

STATIC = Path(__file__).parent / "static"
MAX_QUERY_CHARS = 1000


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS)
    mode: str | None = None
    k: int = Field(5, ge=1, le=20)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUERY_CHARS)
    mode: str | None = None
    k: int | None = Field(None, ge=1, le=10)


def _check_mode(mode: str | None) -> None:
    if mode is not None and mode not in MODES:
        raise HTTPException(422, f"mode must be one of {list(MODES)}")


def create_app(pipeline) -> FastAPI:
    """`pipeline` is a RAGPipeline; tests pass one built on fakes."""
    app = FastAPI(title="Ask the ROS 2 docs", version="1.0.0")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "chunks": len(pipeline.retriever.chunks),
            "default_mode": pipeline.mode,
            "top_k": pipeline.top_k,
            "llm_enabled": pipeline.generator.enabled,
            "llm_model": pipeline.generator.model if pipeline.generator.enabled else None,
        }

    @app.post("/search")
    def search(req: SearchRequest) -> dict:
        _check_mode(req.mode)
        res = pipeline.retriever.search(req.query.strip(), req.mode or pipeline.mode, req.k)
        return {
            "query": req.query,
            "timings_ms": {k: round(v, 1) for k, v in res.timings_ms.items()},
            "results": [
                {"rank": i, "doc_id": h.chunk.doc_id, "section": h.chunk.section,
                 "source": h.chunk.source, "page": h.chunk.page, "score": round(h.score, 4),
                 "ranks": h.ranks, "text": h.chunk.text}
                for i, h in enumerate(res.hits, start=1)
            ],
        }

    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        _check_mode(req.mode)
        try:
            ans = pipeline.ask(req.question.strip(), req.mode, req.k)
        except Exception as exc:  # surface API/network failures as 502, not a stack trace
            if type(exc).__module__.startswith("anthropic"):
                raise HTTPException(502, f"LLM call failed: {type(exc).__name__}") from exc
            raise
        return ans.to_dict()

    return app


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    from roborag.config import load_config
    from roborag.generate import pipeline_from_config

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    app = create_app(pipeline_from_config(load_config(args.config)))
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
