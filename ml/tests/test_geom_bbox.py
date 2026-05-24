"""T-13: Geometry bounding-box tests for raw.rodent_inspections.

Verifies that all non-NULL point geometries in raw.rodent_inspections
lie within the NYC WGS84 bounding box defined in SPEC §10.1:

    lon: -74.26 … -73.68
    lat:  40.49 …  40.92

Any geometry outside this box indicates a coordinate-projection bug
(e.g. NY State Plane EPSG:2263 values stored instead of WGS84) or a
data-cleaning failure in ingest_rodent_inspections.py.

All tests are integration tests (pytest.mark.integration) that require
DATABASE_URL and data loaded by ingest_rodent_inspections.py.
"""

from __future__ import annotations

import os

import pytest

# NYC bounding box per SPEC §10.1
NYC_MINX = -74.26
NYC_MAXX = -73.68
NYC_MINY = 40.49
NYC_MAXY = 40.92


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
async def test_rodent_inspection_geoms_within_nyc_bbox(db_conn) -> None:
    """All non-NULL geoms in raw.rodent_inspections must fall inside the NYC bbox.

    Uses the exact SQL predicate from SPEC §10.1:
        ST_X(geom) BETWEEN -74.26 AND -73.68
        ST_Y(geom) BETWEEN  40.49 AND  40.92
    """
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
        LIMIT 20
        """,
        NYC_MINX, NYC_MAXX, NYC_MINY, NYC_MAXY,
    )
    assert not rows, (
        f"{len(rows)} rodent inspection geometries fall outside NYC bbox "
        f"(lon {NYC_MINX}…{NYC_MAXX}, lat {NYC_MINY}…{NYC_MAXY}). "
        f"Sample: {[dict(r) for r in rows[:5]]}"
    )


@pytest.mark.integration
async def test_rodent_inspections_has_rows(db_conn) -> None:
    """raw.rodent_inspections must be non-empty (ingest ran at least once)."""
    n = await db_conn.fetchval("SELECT COUNT(*) FROM raw.rodent_inspections")
    assert n and n > 0, "raw.rodent_inspections is empty — run ingest_rodent_inspections.py first"


@pytest.mark.integration
async def test_rodent_inspections_geom_coverage(db_conn) -> None:
    """At least 80 % of rodent inspection rows must have a non-NULL geometry.

    A very low geom fill rate (< 80 %) indicates systematic lat/lon data loss
    during ingest — likely a field name mismatch in the Socrata response.
    """
    row = await db_conn.fetchrow(
        """
        SELECT
            COUNT(*)                              AS total,
            COUNT(*) FILTER (WHERE geom IS NULL)  AS null_geom
        FROM raw.rodent_inspections
        """
    )
    total = int(row["total"])
    null_geom = int(row["null_geom"])
    if total == 0:
        pytest.skip("Table is empty")
    null_pct = null_geom / total * 100
    assert null_pct < 20, (
        f"{null_pct:.1f}% of rodent inspection rows have NULL geometry "
        f"({null_geom:,} / {total:,}). Expected < 20%."
    )
