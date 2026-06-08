"""pgvector upsert for RAG corpus chunks (T-25).

Writes to ``app.health_code_chunks`` using asyncpg.
Idempotent on ``chunk_id`` (UUID5 deterministic from content hash).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import asyncpg

if TYPE_CHECKING:
    from rat_ml.rag.corpus import CorpusChunk

log = logging.getLogger(__name__)

_UPSERT_SQL = """
    INSERT INTO app.health_code_chunks (
        chunk_id, document, citation, authority, section_path,
        content, content_with_prefix, token_count,
        version_hash, embedding, effective_date
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::vector, $11)
    ON CONFLICT (chunk_id) DO UPDATE SET
        content              = EXCLUDED.content,
        content_with_prefix  = EXCLUDED.content_with_prefix,
        token_count          = EXCLUDED.token_count,
        version_hash         = EXCLUDED.version_hash,
        embedding            = EXCLUDED.embedding::vector,
        effective_date       = EXCLUDED.effective_date
"""

_UPDATE_BGE_SQL = """
    UPDATE app.health_code_chunks
    SET embedding_bge = $2::vector
    WHERE chunk_id = $1
"""


async def upsert_chunks(
    chunks: "list[CorpusChunk]",
    embeddings: list[list[float]],
    *,
    db_url: str,
    bge_embeddings: list[list[float]] | None = None,
) -> int:
    """Upsert *chunks* with their *embeddings* into ``app.health_code_chunks``.

    Args:
        chunks:         Corpus chunks (from :func:`~rat_ml.rag.corpus.build_corpus`).
        embeddings:     Parallel list of 1024-dim Voyage float vectors.
        db_url:         asyncpg-compatible connection string.
        bge_embeddings: Optional BGE-M3 vectors for ``embedding_bge`` column.

    Returns:
        Number of rows upserted.
    """
    assert len(chunks) == len(embeddings), "chunks and embeddings must be same length"
    if bge_embeddings is not None:
        assert len(chunks) == len(bge_embeddings), "bge_embeddings length mismatch"

    conn = await asyncpg.connect(db_url)
    try:
        async with conn.transaction():
            count = 0
            for i, (chunk, vec) in enumerate(zip(chunks, embeddings, strict=True)):
                vec_str = "[" + ",".join(f"{v:.8f}" for v in vec) + "]"
                await conn.execute(
                    _UPSERT_SQL,
                    chunk.chunk_id,
                    chunk.document,
                    chunk.citation,
                    chunk.authority,
                    chunk.section_path,
                    chunk.content,
                    chunk.content_with_prefix,
                    chunk.token_count,
                    chunk.version_hash,
                    vec_str,
                    chunk.effective_date,
                )
                if bge_embeddings is not None:
                    bge_str = "[" + ",".join(f"{v:.8f}" for v in bge_embeddings[i]) + "]"
                    await conn.execute(_UPDATE_BGE_SQL, chunk.chunk_id, bge_str)
                count += 1
    finally:
        await conn.close()

    log.info("Upserted %d chunks into app.health_code_chunks", count)
    return count
