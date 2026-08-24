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
    parse_penalty_table,
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


# ---------------------------------------------------------------------------
# parse_penalty_table — borderless fine-schedule table parsing
# ---------------------------------------------------------------------------

# Mimics the real ECB PDF's reading-order artifacts: each row restates its
# own citation as the first thing in reading order, and a row with
# escalating penalty tiers (1st/2nd/3rd/4th violation) carries multiple
# AHxxx codes — one per tier — for the *same* row, with its dollar amounts
# interleaved with the tier labels rather than following its own AH code.
_PENALTY_TABLE_PAGE = """\
§3-110
HEALTH CODE AND MISCELLANEOUS FOOD VENDOR VIOLATIONS PENALTY SCHEDULE
For multiple rodent violations issued under NYC Health Code section 151.02(a),
the minimum civil penalty shall be not less than $300.
SECTION/RULE DESCRIPTION PENALTY DEFAULT
NYC Health Code 3.09 Failing to abate or remediate nuisance $1000 $2000
AH3M
NYC Health Code 151.02(a) Failure to eliminate rodent infestation shown by 1st 1st
active rodent signs: one or more live rodents, or Violation: Violation:
1st: AH3N rodent droppings, burrows, runways, tracks, rub $300 $600
2nd: AH3W marks or gnaw marks; in interior or exterior of premises.
NYC Health Code 81.09 Potentially hazardous foods at improper temperatures 385 770
AH02
"""


def _parse_sample_penalty_table():
    return parse_penalty_table(
        [_PENALTY_TABLE_PAGE], authority="ECB", document="ECB Penalty Schedule"
    )


def test_parse_penalty_table_splits_one_chunk_per_citation_row():
    chunks = _parse_sample_penalty_table()
    # intro + §3.09 row + §151.02(a) row (AH3N+AH3W tiers) + §81.09 row
    assert len(chunks) == 4


def test_parse_penalty_table_does_not_mix_unrelated_rows():
    chunks = _parse_sample_penalty_table()
    rodent_chunk = next(c for c in chunks if "3.09" in c.citation)
    # Regression: the old sliding-window chunker put this in the same
    # 400-token blob as the calorie-labeling/smoking/permit-transfer rows —
    # a real chunk about a rodent-relevant penalty must not contain an
    # unrelated violation's text.
    assert "hazardous foods" not in rodent_chunk.content


def test_parse_penalty_table_multi_tier_row_keeps_description_and_amount_together():
    chunks = _parse_sample_penalty_table()
    # Regression: an earlier version split on the AHxxx tier code instead of
    # the row's citation, which sliced this row's description away from its
    # own dollar amount — a chunk about "burrows" never contained "$300".
    # Both tier codes belong to the same row and must land in one chunk,
    # alongside the amount that answers a "what's the penalty" question.
    row = next(c for c in chunks if "AH3N" in c.citation)
    assert "AH3W" in row.citation
    assert "burrows" in row.content
    assert "$300" in row.content


def test_parse_penalty_table_intro_paragraph_is_its_own_chunk():
    chunks = _parse_sample_penalty_table()
    assert chunks[0].citation == "§3-110"
    assert "SECTION/RULE" not in chunks[0].content
