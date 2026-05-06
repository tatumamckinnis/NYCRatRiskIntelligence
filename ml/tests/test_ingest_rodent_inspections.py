"""Tests for ingest_rodent_inspections.

Unit tests: row parsing logic, no DB or Socrata required.
Integration tests (pytest.mark.integration): idempotency and geom bbox,
require DATABASE_URL and a live Supabase connection.
"""

from __future__ import annotations

import os

import pytest

from rat_ml.data.ingest_rodent_inspections import _parse_row, upsert_batch

# NYC bounding box (WGS84) — same values as test_crosswalk.py
NYC_MINX, NYC_MAXX = -74.30, -73.65
NYC_MINY, NYC_MAXY = 40.40, 40.95


# ===========================================================================
# Unit tests — _parse_row
# ===========================================================================

SAMPLE_ROW = {
    "inspectionid": "INS-001",
    "inspectiondate": "2024-03-15T00:00:00.000",
    "bbl": "3007390001",
    "bin": "3012345",
    "boro": "3",
    "block": "739",
    "lot": "1",
    "result": "Active Rat Signs",
    "inspection_type": "Initial",
    "jobprogress": "2",
    "latitude": "40.6892",
    "longitude": "-73.9442",
}


def test_parse_valid_row() -> None:
    p = _parse_row(SAMPLE_ROW)
    assert p is not None
    assert p["inspection_id"] == "INS-001"
    assert p["bbl"] == "3007390001"
    assert p["result"] == "Active Rat Signs"
    assert p["geom_wkt"] == "POINT(-73.9442 40.6892)"
    assert not p["raw_bbl_missing"]


def test_parse_missing_inspection_id_returns_none() -> None:
    row = {**SAMPLE_ROW, "inspectionid": None}
    assert _parse_row(row) is None


def test_parse_invalid_bbl_flags_unmatched() -> None:
    row = {**SAMPLE_ROW, "bbl": "NOT_A_BBL"}
    p = _parse_row(row)
    assert p is not None
    assert p["bbl"] is None
    assert p["raw_bbl_missing"]


def test_parse_null_bbl_not_flagged_unmatched() -> None:
    # None BBL is expected missing, not a parse error — don't flag it.
    row = {**SAMPLE_ROW, "bbl": None}
    p = _parse_row(row)
    assert p is not None
    assert not p["raw_bbl_missing"]


def test_parse_no_coordinates_gives_null_geom() -> None:
    row = {**SAMPLE_ROW, "latitude": None, "longitude": None}
    p = _parse_row(row)
    assert p is not None
    assert p["geom_wkt"] is None


def test_parse_invalid_borough_gives_none() -> None:
    row = {**SAMPLE_ROW, "boro": "X"}
    p = _parse_row(row)
    assert p is not None
    assert p["borough"] is None


def test_parse_integer_bbl() -> None:
    row = {**SAMPLE_ROW, "bbl": 3007390001}
    p = _parse_row(row)
    assert p is not None
    assert p["bbl"] == "3007390001"


# ===========================================================================
# Integration tests
# ===========================================================================

@pytest.fixture()
async def db_conn():
    import asyncpg  # noqa: PLC0415
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    conn = await asyncpg.connect(url)
    try:
        yield conn
    finally:
        await conn.close()


@pytest.mark.integration
async def test_idempotency_rodent_inspections(db_conn) -> None:  # type: ignore[misc]
    """Running upsert_batch twice with the same rows must not change the row count."""
    from rat_ml.data._socrata import get_client, paginate  # noqa: PLC0415
    from rat_ml.data.ingest_rodent_inspections import _cutoff_date  # noqa: PLC0415

    client = get_client()
    # Fetch only the first page (up to 50k rows) for speed.
    first_batch: list[dict] = []
    for batch, _ in paginate(
        client,
        "p937-wjvj",
        where=f"inspectiondate >= '{_cutoff_date()}'",
        order=":id",
    ):
        first_batch = batch
        break

    if not first_batch:
        pytest.skip("No data returned from Socrata — check token or dataset availability")

    await upsert_batch(db_conn, first_batch)
    count_after_first = await db_conn.fetchval("SELECT COUNT(*) FROM raw.rodent_inspections")

    await upsert_batch(db_conn, first_batch)
    count_after_second = await db_conn.fetchval("SELECT COUNT(*) FROM raw.rodent_inspections")

    assert count_after_first == count_after_second, (
        f"Row count changed on second upsert: {count_after_first} → {count_after_second}"
    )


@pytest.mark.integration
async def test_geom_bbox_rodent_inspections(db_conn) -> None:  # type: ignore[misc]
    """All non-NULL geometries in raw.rodent_inspections must lie within the NYC bbox."""
    rows = await db_conn.fetch(
        """
        SELECT inspection_id,
               ST_X(geom) AS lon,
               ST_Y(geom) AS lat
        FROM raw.rodent_inspections
        WHERE geom IS NOT NULL
          AND NOT (
              ST_X(geom) BETWEEN $1 AND $2
              AND ST_Y(geom) BETWEEN $3 AND $4
          )
        LIMIT 10
        """,
        NYC_MINX, NYC_MAXX, NYC_MINY, NYC_MAXY,
    )
    assert not rows, (
        f"{len(rows)} geometries outside NYC bbox. "
        f"Sample: {[dict(r) for r in rows[:3]]}"
    )
