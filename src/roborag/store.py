"""Vector store: Chroma (persistent, HNSW, cosine) holding chunk embeddings + metadata."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from roborag.chunking import Chunk


def collection_name(strategy: str, model_name: str) -> str:
    slug = model_name.split("/")[-1].lower().replace(".", "-")
    return f"{strategy}-{slug}"[:63]


class ChromaStore:
    def __init__(self, path: Path | str, name: str):
        import chromadb
        from chromadb.config import Settings

        self.client = chromadb.PersistentClient(path=str(path),
                                                settings=Settings(anonymized_telemetry=False))
        self.name = name
        self.collection = self.client.get_or_create_collection(
            name, metadata={"hnsw:space": "cosine"}
        )

    def reset(self) -> None:
        self.client.delete_collection(self.name)
        self.collection = self.client.create_collection(self.name,
                                                        metadata={"hnsw:space": "cosine"})

    def count(self) -> int:
        return self.collection.count()

    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray, batch: int = 2000) -> None:
        for i in range(0, len(chunks), batch):
            part = chunks[i : i + batch]
            self.collection.add(
                ids=[c.chunk_id for c in part],
                embeddings=embeddings[i : i + batch].tolist(),
                documents=[c.text for c in part],
                metadatas=[{"doc_id": c.doc_id, "section": c.section, "source": c.source,
                            "page": c.page or 0} for c in part],
            )

    def query(self, embedding: np.ndarray, n: int) -> list[tuple[str, float]]:
        """Return [(chunk_id, cosine similarity)] best first."""
        res = self.collection.query(query_embeddings=[embedding.tolist()], n_results=n,
                                    include=["distances"])
        return [(cid, 1.0 - float(d)) for cid, d in zip(res["ids"][0], res["distances"][0],
                                                          strict=True)]


def save_chunks(chunks: Sequence[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
