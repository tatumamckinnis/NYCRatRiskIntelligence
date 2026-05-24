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
