"""Thin wrappers around the two local neural models: the bi-encoder and the cross-encoder.

Both pick Apple's GPU (`mps`) when available and fall back to CPU.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    """Bi-encoder: query and passage are embedded *separately*, so passages can be indexed once."""

    def __init__(self, model_name: str, query_prefix: str = "", batch_size: int = 64,
                 device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device or pick_device())

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def _encode(self, texts: Sequence[str], progress: bool = False) -> np.ndarray:
        emb = self.model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True,
                                show_progress_bar=progress, convert_to_numpy=True)
        return emb.astype(np.float32)

    def encode_docs(self, texts: Sequence[str], progress: bool = False) -> np.ndarray:
        return self._encode(texts, progress)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        # BGE models were trained with an instruction on the query side only
        return self._encode([self.query_prefix + t for t in texts])


class Reranker:
    """Cross-encoder: reads query and passage *together* and outputs one relevance score.

    Far more accurate than comparing two independent embeddings, but it must run once per
    (query, passage) pair, so it only re-scores the few dozen candidates retrieval returns.
    """

    def __init__(self, model_name: str, device: str | None = None, max_length: int = 512):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device or pick_device(),
                                  max_length=max_length)

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        scores = self.model.predict([(query, t) for t in texts], batch_size=32,
                                    show_progress_bar=False)
        return [float(s) for s in scores]
