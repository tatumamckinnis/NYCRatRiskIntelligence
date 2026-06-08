"""BGE Reranker v2-M3 (T-39).

Loads ``BAAI/bge-reranker-v2-m3`` via ``sentence-transformers.CrossEncoder``
at API startup (via the lifespan dependency).  Falls back to Cohere Rerank 3.5
if ``COHERE_API_KEY`` is set (ablation path only).

The reranker is intended to be loaded once and reused across requests.

Usage::

    from rat_api.rag.reranker import BgeReranker, get_reranker

    reranker = get_reranker()                    # singleton
    ranked = reranker.rerank(query, chunks, top_k=6)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rat_api.rag.retriever import RetrievedChunk

log = logging.getLogger(__name__)

BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Module-level singleton
_reranker: "BgeReranker | None" = None


@dataclass
class BgeReranker:
    """Cross-encoder reranker backed by BGE Reranker v2-M3.

    Args:
        model_name: HuggingFace model ID.
        device:     ``"cpu"``, ``"cuda"``, or ``"mps"``; ``None`` = auto.
    """

    model_name: str = BGE_RERANKER_MODEL
    device: str | None = None

    def __post_init__(self) -> None:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415
        log.info("Loading BGE Reranker %s …", self.model_name)
        self._model = CrossEncoder(self.model_name, device=self.device)
        log.info("BGE Reranker loaded.")

    def rerank(
        self,
        query: str,
        chunks: "list[RetrievedChunk]",
        *,
        top_k: int = 6,
    ) -> "list[RetrievedChunk]":
        """Score (query, chunk) pairs and return the top *top_k* by score.

        Args:
            query:  User query string.
            chunks: Candidate chunks to rerank.
            top_k:  Number of chunks to return.

        Returns:
            List of up to *top_k* chunks sorted by descending rerank score,
            with ``chunk.score`` updated to the cross-encoder score.
        """
        if not chunks:
            return []
        pairs = [(query, c.content) for c in chunks]
        scores = self._model.predict(pairs)  # type: ignore[arg-type]
        ranked = sorted(
            zip(scores, chunks),
            key=lambda x: float(x[0]),
            reverse=True,
        )
        result = []
        for score, chunk in ranked[:top_k]:
            from dataclasses import replace  # noqa: PLC0415
            result.append(replace(chunk, score=float(score)))
        return result


def load_reranker(device: str | None = None) -> BgeReranker:
    """Construct and return a :class:`BgeReranker` (called at lifespan startup)."""
    global _reranker  # noqa: PLW0603
    _reranker = BgeReranker(device=device)
    return _reranker


def get_reranker() -> "BgeReranker | None":
    """Return the singleton reranker (None if not yet loaded)."""
    return _reranker
