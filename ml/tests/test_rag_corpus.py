"""Unit tests for RAG corpus builder (T-29)."""

from __future__ import annotations

import pytest

from rat_ml.rag.corpus import CorpusChunk, build_corpus, iter_static_chunks


def test_build_corpus_returns_chunks():
    chunks = build_corpus()
    assert len(chunks) > 0


def test_all_chunks_have_required_fields():
    for chunk in build_corpus():
        assert chunk.chunk_id, "chunk_id must not be empty"
        assert chunk.document, "document must not be empty"
        assert chunk.content, "content must not be empty"
        assert chunk.content_with_prefix, "content_with_prefix must not be empty"
        assert chunk.token_count > 0, "token_count must be positive"
        assert len(chunk.version_hash) == 64, "version_hash should be sha256 hex (64 chars)"


def test_chunk_ids_are_unique():
    chunks = build_corpus()
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_ids must be unique"


def test_content_with_prefix_contains_content():
    for chunk in build_corpus():
        # prefix format is "From <authority> — <citation>: <content>"
        assert chunk.content in chunk.content_with_prefix


def test_three_documents_present():
    docs = {c.document for c in build_corpus()}
    assert "nyc_health_code_art151" in docs
    assert "dsny_containerization_policy" in docs
    assert "dohmh_rodent_mitigation" in docs


def test_static_chunks_are_deterministic():
    """Same input → same chunk_ids (UUID5 is deterministic)."""
    ids_first = [c.chunk_id for c in iter_static_chunks()]
    ids_second = [c.chunk_id for c in iter_static_chunks()]
    assert ids_first == ids_second


def test_chunk_is_dataclass():
    chunk = next(iter_static_chunks())
    assert isinstance(chunk, CorpusChunk)
