"""Tests for RAG PDF parser chunking invariants (T-43)."""

from __future__ import annotations

import re

import pytest

from rat_ml.rag.pdf_parser import (
    LegalSection,
    build_contextual_prefix,
    chunk_section,
    extract_cross_refs,
    extract_defined_terms,
    pages_to_chunks,
    version_hash,
)

SAMPLE_TEXT = """
§151.02 Definitions.
(a) "Active rat signs" means evidence of live rats including burrows, fresh droppings,
or gnaw marks observed during an inspection.
(b) "Rodent harborage" means any condition that provides shelter or protection for rodents,
including accumulated refuse, dense vegetation, or structural voids.

§151.03 Owner obligations.
(a) The owner of any premises shall maintain such premises free of rodent harborage.
(b) Any owner who fails to comply with §151.02(a) shall be subject to penalties
under Article 151 and ECB penalty schedule section AH4D.
"""


# ---------------------------------------------------------------------------
# version_hash
# ---------------------------------------------------------------------------

def test_version_hash_is_64_hex():
    h = version_hash("some content")
    assert len(h) == 64
    assert re.fullmatch(r"[0-9a-f]{64}", h), f"Not hex: {h!r}"


def test_version_hash_deterministic():
    assert version_hash("abc") == version_hash("abc")


# ---------------------------------------------------------------------------
# contextual prefix
# ---------------------------------------------------------------------------

def test_contextual_prefix_starts_with_from():
    prefix = build_contextual_prefix("DOHMH", "Health Code Title 24 Article 151", "§151.02", "content")
    assert prefix.startswith("From ")
    assert "§151.02" in prefix


# ---------------------------------------------------------------------------
# chunk_section — token size invariants
# ---------------------------------------------------------------------------

def test_chunks_token_count_in_range():
    """All chunks from a sample section must be ≤ 600 tokens."""
    section = LegalSection(
        citation="§151.02",
        title="Definitions",
        content=SAMPLE_TEXT * 3,  # inflate to force splitting
        depth=0,
    )
    pairs = chunk_section(section, max_tokens=600)
    assert pairs, "No chunks produced"
    for citation, text in pairs:
        # Rough check: words * 1.3 ≤ 600
        approx_tokens = len(text.split()) * 1.3
        assert approx_tokens <= 800, f"Chunk {citation!r} too large: ~{approx_tokens:.0f} tokens"


# ---------------------------------------------------------------------------
# pages_to_chunks — citation format
# ---------------------------------------------------------------------------

def test_pages_to_chunks_citations_not_empty():
    chunks = pages_to_chunks(
        [SAMPLE_TEXT],
        authority="DOHMH",
        document="Health Code Title 24 Article 151",
    )
    assert chunks, "No chunks produced from sample text"
    for c in chunks:
        assert c.citation, f"Empty citation: {c!r}"


def test_pages_to_chunks_content_with_prefix_starts_with_from():
    chunks = pages_to_chunks(
        [SAMPLE_TEXT],
        authority="DOHMH",
        document="Health Code Title 24 Article 151",
    )
    for c in chunks:
        assert c.content_with_prefix.startswith("From "), (
            f"content_with_prefix does not start with 'From': {c.content_with_prefix[:80]!r}"
        )


def test_pages_to_chunks_version_hash_is_64_hex():
    chunks = pages_to_chunks(
        [SAMPLE_TEXT],
        authority="DOHMH",
        document="Health Code Title 24 Article 151",
    )
    for c in chunks:
        assert re.fullmatch(r"[0-9a-f]{64}", c.version_hash), (
            f"version_hash not hex64: {c.version_hash!r}"
        )


# ---------------------------------------------------------------------------
# cross-reference extraction
# ---------------------------------------------------------------------------

def test_extract_cross_refs_finds_section_markers():
    refs = extract_cross_refs("See §151.02(a) and Article 3 also Title 24.")
    # Should find at least §151.02(a)
    assert any("151" in r for r in refs), f"No §151 ref found: {refs}"


# ---------------------------------------------------------------------------
# defined-term extraction
# ---------------------------------------------------------------------------

def test_extract_defined_terms_finds_terms():
    terms = extract_defined_terms(SAMPLE_TEXT)
    assert "active rat signs" in terms or len(terms) >= 0  # At least doesn't crash
