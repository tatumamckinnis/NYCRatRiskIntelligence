"""Voyage AI embedding for RAG corpus chunks (T-25).

Uses ``voyage-3`` (1024-dim) — the same model wired to ``app.health_code_chunks``.
Batches automatically to stay within the Voyage AI rate limit (128 docs / request).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rat_ml.rag.corpus import CorpusChunk

log = logging.getLogger(__name__)

VOYAGE_MODEL = "voyage-3"
VOYAGE_BATCH_SIZE = 128
VOYAGE_INPUT_TYPE = "document"


def embed_chunks(
    chunks: "list[CorpusChunk]",
    *,
    api_key: str,
    model: str = VOYAGE_MODEL,
    batch_size: int = VOYAGE_BATCH_SIZE,
    sleep_between_batches: float = 0.5,
) -> list[list[float]]:
    """Embed *chunks* using Voyage AI and return a list of float vectors.

    Args:
        chunks:                  Corpus chunks to embed (uses ``content_with_prefix``).
        api_key:                 Voyage AI API key.
        model:                   Voyage model name.
        batch_size:              Documents per API call (max 128).
        sleep_between_batches:   Seconds to sleep between batches (rate-limit guard).

    Returns:
        List of 1024-dim float vectors, one per chunk, in input order.
    """
    import voyageai  # noqa: PLC0415

    client = voyageai.Client(api_key=api_key)
    texts = [c.content_with_prefix for c in chunks]
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        log.info(
            "Embedding batch %d–%d of %d …",
            i + 1,
            min(i + batch_size, len(texts)),
            len(texts),
        )
        result = client.embed(batch, model=model, input_type=VOYAGE_INPUT_TYPE)
        all_embeddings.extend(result.embeddings)

        if i + batch_size < len(texts):
            time.sleep(sleep_between_batches)

    log.info("Embedded %d chunks → %d vectors", len(chunks), len(all_embeddings))
    return all_embeddings


def embed_query(
    query: str,
    *,
    api_key: str,
    model: str = VOYAGE_MODEL,
) -> list[float]:
    """Embed a single retrieval query string.

    Uses ``input_type="query"`` which Voyage optimises for retrieval.
    """
    import voyageai  # noqa: PLC0415

    client = voyageai.Client(api_key=api_key)
    result = client.embed([query], model=model, input_type="query")
    return result.embeddings[0]


# ---------------------------------------------------------------------------
# BGE-M3 ablation embedder (T-38)
# ---------------------------------------------------------------------------

BGE_M3_MODEL = "BAAI/bge-m3"
BGE_BATCH_SIZE = 128


class BgeMThreeEmbedder:
    """Self-hosted BGE-M3 embedder for the ablation column (``embedding_bge``).

    Loads the model once at initialisation.  Uses ``sentence-transformers``
    ``encode()`` which handles batching and normalisation internally.

    Args:
        model_name: HuggingFace model ID (default: ``BAAI/bge-m3``).
        device:     ``"cpu"``, ``"cuda"``, or ``"mps"``; ``None`` = auto.
    """

    def __init__(self, model_name: str = BGE_M3_MODEL, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        log.info("Loading BGE-M3 model %s …", model_name)
        self._model = SentenceTransformer(model_name, device=device)
        log.info("BGE-M3 loaded.")

    def encode(self, texts: list[str], *, batch_size: int = BGE_BATCH_SIZE) -> list[list[float]]:
        """Return one 1024-dim float vector per text."""
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]


_bge_singleton: BgeMThreeEmbedder | None = None


def _get_bge_embedder() -> BgeMThreeEmbedder:
    global _bge_singleton
    if _bge_singleton is None:
        # Force CPU — MPS runs out of memory on large batches (>~100 chunks).
        _bge_singleton = BgeMThreeEmbedder(device="cpu")
    return _bge_singleton


def embed_query_bge(query: str) -> list[float]:
    """Embed a single retrieval query string using the local BGE-M3 model (free).

    The model is loaded once and cached as a module-level singleton.
    """
    return _get_bge_embedder().encode([query])[0]


def embed_chunks_bge(
    chunks: "list[CorpusChunk]",
    *,
    embedder: "BgeMThreeEmbedder | None" = None,
    device: str | None = None,
    batch_size: int = BGE_BATCH_SIZE,
) -> list[list[float]]:
    """Embed *chunks* with BGE-M3 and return 1024-dim float vectors.

    Args:
        chunks:    Corpus chunks; uses ``content_with_prefix`` as the text.
        embedder:  Pre-constructed :class:`BgeMThreeEmbedder`; created if None.
        device:    Torch device override.
        batch_size: Sentences per forward pass.

    Returns:
        List of 1024-dim float vectors, one per chunk.
    """
    if embedder is None:
        embedder = _get_bge_embedder() if device is None else BgeMThreeEmbedder(device=device)
    texts = [c.content_with_prefix for c in chunks]
    vecs = embedder.encode(texts, batch_size=batch_size)
    log.info("BGE-M3 embedded %d chunks", len(chunks))
    return vecs
