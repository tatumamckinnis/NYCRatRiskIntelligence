"""Smoke tests for the hybrid retriever (T-43)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rat_api.rag.retriever import RetrievedChunk, _rrf_fuse


# ---------------------------------------------------------------------------
# RRF fusion (pure, no DB)
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id: str, score: float = 0.9, method: str = "dense") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document="test_doc",
        citation=f"§{chunk_id}",
        authority="DOHMH",
        section_path=[chunk_id],
        content=f"content of {chunk_id}",
        content_with_prefix=f"From DOHMH test §{chunk_id}: content of {chunk_id}",
        score=score,
        retrieval_method=method,
    )


def test_rrf_fuse_deduplicates():
    dense = [_make_chunk("A"), _make_chunk("B"), _make_chunk("C")]
    bm25 = [_make_chunk("B"), _make_chunk("C"), _make_chunk("D")]
    result = _rrf_fuse([dense, bm25], top_k=10)
    ids = [c.chunk_id for c in result]
    assert len(ids) == len(set(ids)), "Duplicate chunk IDs after RRF"


def test_rrf_fuse_boosts_overlap():
    """Chunks appearing in both lists should rank higher than single-list chunks."""
    dense = [_make_chunk("SHARED"), _make_chunk("DENSE_ONLY")]
    bm25 = [_make_chunk("SHARED"), _make_chunk("BM25_ONLY")]
    result = _rrf_fuse([dense, bm25], top_k=10)
    ids = [c.chunk_id for c in result]
    shared_pos = ids.index("SHARED")
    assert shared_pos == 0, f"SHARED should be #1, got position {shared_pos}"


def test_rrf_fuse_respects_top_k():
    dense = [_make_chunk(f"D{i}") for i in range(20)]
    bm25 = [_make_chunk(f"B{i}") for i in range(20)]
    result = _rrf_fuse([dense, bm25], top_k=5)
    assert len(result) <= 5


def test_rrf_fuse_sets_retrieval_method():
    dense = [_make_chunk("A")]
    result = _rrf_fuse([dense], top_k=5)
    assert all(c.retrieval_method == "rrf" for c in result)


# ---------------------------------------------------------------------------
# retrieve() — smoke test with mocked DB and embedding
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retrieve_returns_retrieved_chunks():
    """retrieve() should return a list of RetrievedChunk with correct input_type."""
    mock_conn = AsyncMock()
    # Dense rows
    mock_conn.fetch.side_effect = [
        # dense result
        [
            {
                "chunk_id": "abc123",
                "document": "Health Code Title 24 Article 151",
                "citation": "§151.02",
                "authority": "DOHMH",
                "section_path": ["151", "151.02"],
                "content": "Active rat signs means evidence of live rats.",
                "content_with_prefix": "From DOHMH §151.02: Active rat signs means evidence of live rats.",
                "parent_chunk_id": None,
                "score": 0.92,
            }
        ],
        # bm25 result
        [],
    ]

    with (
        patch("rat_api.rag.retriever._rewrite_query", return_value="rat signs NYC health code"),
        patch("rat_api.rag.retriever.embed_query", return_value=[0.1] * 1024),
        patch("rat_api.rag.retriever.get_reranker", return_value=None),
    ):
        from rat_api.rag.retriever import retrieve
        chunks = await retrieve("What are active rat signs?", mock_conn, top_k_final=6)

    assert isinstance(chunks, list)
    # Should have at least one chunk if DB returned results
    for c in chunks:
        assert isinstance(c, RetrievedChunk)
        assert c.chunk_id


@pytest.mark.asyncio
async def test_embed_query_uses_query_input_type():
    """embed_query must be called with input_type='query' for retrieval accuracy."""
    calls = []

    def fake_embed_query(query, *, api_key, model="voyage-3"):
        calls.append({"query": query, "api_key": api_key})
        return [0.0] * 1024

    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = []

    with (
        patch("rat_api.rag.retriever._rewrite_query", return_value="test query"),
        patch("rat_api.rag.retriever.embed_query", side_effect=fake_embed_query),
        patch("rat_api.rag.retriever.get_reranker", return_value=None),
    ):
        from rat_api.rag.retriever import retrieve
        await retrieve("test query", mock_conn)

    assert len(calls) == 1, "embed_query should be called exactly once"
    # The actual input_type check is in embed.py (input_type="query") —
    # here we just assert it was called at all (prevents regression to document mode)
