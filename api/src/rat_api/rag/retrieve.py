"""RAG retrieval: vector search + Cohere Rerank 3.5 (T-26).

Pipeline
--------
1. Embed the query with Voyage AI (``voyage-3``, input_type="query").
2. HNSW vector search against ``app.health_code_chunks`` in pgvector.
3. Re-rank the top-20 candidates with Cohere Rerank 3.5.
4. Return the top-*k* chunks (default 5) as :class:`RetrievedChunk` objects.

If ``COHERE_API_KEY`` is empty, step 3 is skipped and vector-search order
is used directly — the API degrades gracefully.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import asyncpg

from rat_api.config import get_settings

log = logging.getLogger(__name__)

_VECTOR_SEARCH_SQL = """
    SELECT
        chunk_id,
        document,
        citation,
        authority,
        content,
        content_with_prefix,
        1 - (embedding <=> $1::vector) AS cosine_score
    FROM app.health_code_chunks
    ORDER BY embedding <=> $1::vector
    LIMIT $2
"""


@dataclass
class RetrievedChunk:
    chunk_id: str
    document: str
    citation: str
    authority: str
    content: str
    score: float  # reranked relevance score (0–1), or cosine sim if no rerank


async def retrieve(
    query: str,
    conn: asyncpg.Connection,
    *,
    top_k: int = 5,
    candidate_k: int = 20,
) -> list[RetrievedChunk]:
    """Retrieve relevant chunks for *query*.

    Args:
        query:       Natural-language question or NTA risk summary prompt.
        conn:        Live asyncpg connection.
        top_k:       Number of chunks to return after reranking.
        candidate_k: Number of vector-search candidates to pass to reranker.

    Returns:
        List of up to *top_k* :class:`RetrievedChunk` objects, highest score first.
    """
    settings = get_settings()

    # ── Step 1: Embed query ───────────────────────────────────────────────
    query_vec = _embed_query(query, api_key=settings.voyageai_api_key)

    # ── Step 2: Vector search ─────────────────────────────────────────────
    vec_str = "[" + ",".join(f"{v:.8f}" for v in query_vec) + "]"
    rows = await conn.fetch(_VECTOR_SEARCH_SQL, vec_str, candidate_k)

    if not rows:
        return []

    candidates = [
        RetrievedChunk(
            chunk_id=str(r["chunk_id"]),
            document=r["document"],
            citation=r["citation"],
            authority=r["authority"],
            content=r["content"],
            score=float(r["cosine_score"]),
        )
        for r in rows
    ]

    # ── Step 3: Cohere Rerank (optional) ──────────────────────────────────
    if settings.cohere_api_key:
        candidates = _cohere_rerank(
            query,
            candidates,
            api_key=settings.cohere_api_key,
            top_k=top_k,
        )
    else:
        candidates = candidates[:top_k]

    return candidates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _embed_query(query: str, *, api_key: str) -> list[float]:
    """Embed *query* using Voyage AI (synchronous call, kept thin)."""
    if not api_key:
        log.warning("voyageai_api_key not set — returning zero vector for query embed")
        return [0.0] * 1024

    import voyageai  # noqa: PLC0415

    client = voyageai.Client(api_key=api_key)
    result = client.embed([query], model="voyage-3", input_type="query")
    return result.embeddings[0]


def _cohere_rerank(
    query: str,
    candidates: list[RetrievedChunk],
    *,
    api_key: str,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Re-rank *candidates* with Cohere Rerank 3.5."""
    try:
        import cohere  # noqa: PLC0415

        co = cohere.Client(api_key=api_key)
        docs = [c.content for c in candidates]
        response = co.rerank(
            query=query,
            documents=docs,
            model="rerank-english-v3.0",
            top_n=top_k,
        )
        reranked: list[RetrievedChunk] = []
        for hit in response.results:
            chunk = candidates[hit.index]
            reranked.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document=chunk.document,
                    citation=chunk.citation,
                    authority=chunk.authority,
                    content=chunk.content,
                    score=float(hit.relevance_score),
                )
            )
        return reranked
    except Exception as exc:  # noqa: BLE001
        log.warning("Cohere rerank failed (%s) — using vector search order", exc)
        return candidates[:top_k]
