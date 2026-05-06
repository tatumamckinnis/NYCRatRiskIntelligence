"""Tests for NTA boundary data and the 2010→2020 crosswalk.

Unit tests run without a DB. Integration tests (pytest.mark.integration)
require a live Supabase connection via DATABASE_URL and data loaded by
ml/scripts/load_nta_boundaries.py.

NYC WGS84 bounding box used throughout (per SPEC §5.3):
    minx = -74.30, miny = 40.40, maxx = -73.65, maxy = 40.95

The bounding box is intentionally slightly larger than the five-borough
extent to allow for water boundaries and minor DCP geometry artefacts,
while still catching the EPSG:2263 / EPSG:4326 confusion (NY State Plane
X values are in the range 900_000–1_100_000 ft, which would fail this
check immediately).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from rat_ml.data.tract_crosswalk import (
    all_nta2020_codes,
    allocate,
    load_crosswalk,
    weights_for,
)

# ---------------------------------------------------------------------------
# Bounding box constants (SPEC §5.3 + user-specified values for this task)
# ---------------------------------------------------------------------------
NYC_MINX = -74.30
NYC_MINY = 40.40
NYC_MAXX = -73.65
NYC_MAXY = 40.95


# ===========================================================================
# Unit tests — no DB required
# ===========================================================================

SAMPLE_CROSSWALK_CSV = """\
nta2010,nta2020,overlap_pct
BK0101,BK2201,0.6
BK0101,BK2202,0.4
MN2501,MN2501,1.0
QN9901,QN9901,0.5
QN9901,QN9902,0.5
"""


@pytest.fixture()
def loaded_sample(tmp_path: Path) -> Path:
    """Write a minimal crosswalk CSV and load it into the module."""
    csv_path = tmp_path / "crosswalk.csv"
    csv_path.write_text(SAMPLE_CROSSWALK_CSV)
    load_crosswalk(csv_path)
    return csv_path


def test_weights_sum_to_one_sample(loaded_sample: Path) -> None:
    """Weights for each 2010 NTA sum to exactly 1.0 after normalization."""
    for nta2010 in ["BK0101", "MN2501", "QN9901"]:
        w = weights_for(nta2010)
        assert abs(sum(w.values()) - 1.0) < 1e-9, (
            f"Weights for {nta2010} sum to {sum(w.values())}, expected 1.0"
        )


def test_allocate_preserves_value(loaded_sample: Path) -> None:
    """allocate() distributes the full value without gain or loss."""
    value = 12345.67
    result = allocate(value, "BK0101")
    assert abs(sum(result.values()) - value) < 1e-6


def test_allocate_unknown_nta_returns_empty(loaded_sample: Path) -> None:
    assert allocate(100.0, "XX9999") == {}


def test_weights_for_unknown_nta_returns_empty(loaded_sample: Path) -> None:
    assert weights_for("XX9999") == {}


def test_column_name_normalization(tmp_path: Path) -> None:
    """load_crosswalk() accepts alternative column name variants."""
    csv_path = tmp_path / "crosswalk_alt.csv"
    csv_path.write_text("NTA10,NTA20,PCT\nBK0101,BK2201,1.0\n")
    load_crosswalk(csv_path)
    w = weights_for("BK0101")
    assert "BK2201" in w
    assert abs(w["BK2201"] - 1.0) < 1e-9


def test_imperfect_weights_normalized(tmp_path: Path) -> None:
    """Weights that sum to != 1.0 in the source file are normalized."""
    csv_path = tmp_path / "crosswalk_imperfect.csv"
    # Weights sum to 0.9 (common floating-point artefact in area-overlap CSVs)
    csv_path.write_text("nta2010,nta2020,overlap_pct\nBK0101,BK2201,0.54\nBK0101,BK2202,0.36\n")
    load_crosswalk(csv_path)
    w = weights_for("BK0101")
    assert abs(sum(w.values()) - 1.0) < 1e-9


# ===========================================================================
# Integration tests — require live DB + loaded data
# ===========================================================================

@pytest.fixture()
async def db_conn():
    """Yield an asyncpg connection, skip if DATABASE_URL is absent."""
    import asyncpg  # noqa: PLC0415

    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set — skipping integration test")
    conn = await asyncpg.connect(url)
    try:
        yield conn
    finally:
        await conn.close()


@pytest.mark.integration
async def test_all_2020_nta_ids_in_crosswalk(db_conn, tmp_path: Path) -> None:  # type: ignore[misc]
    """Every NTA ID in raw.nta_boundaries must appear in the crosswalk as a 2020 target.

    Ensures load_nta_boundaries.py and the crosswalk cover the same NTA universe
    so that T-06 spatial joins and T-09 adjacency calculations are complete.
    """
    crosswalk_path = Path(os.environ.get("NTA_CROSSWALK_PATH", "data/nta_2010_2020_crosswalk.csv"))
    if not crosswalk_path.exists():
        # Try .xlsx variant
        crosswalk_path = crosswalk_path.with_suffix(".xlsx")
    if not crosswalk_path.exists():
        pytest.skip(f"Crosswalk file not found at {crosswalk_path}")

    if crosswalk_path.suffix == ".xlsx":
        df = pd.read_excel(crosswalk_path)
        df.columns = df.columns.str.lower().str.strip()
        tmp_csv = tmp_path / "crosswalk.csv"
        df.to_csv(tmp_csv, index=False)
        load_crosswalk(tmp_csv)
    else:
        load_crosswalk(crosswalk_path)

    db_nta_ids: set[str] = {
        r["nta_id"] for r in await db_conn.fetch("SELECT nta_id FROM raw.nta_boundaries")
    }
    assert db_nta_ids, "raw.nta_boundaries is empty — run load_nta_boundaries.py first"

    crosswalk_2020_ids = all_nta2020_codes()
    missing = db_nta_ids - crosswalk_2020_ids
    assert not missing, (
        f"{len(missing)} NTA IDs in raw.nta_boundaries have no crosswalk entry: {sorted(missing)}"
    )


@pytest.mark.integration
async def test_allocation_weights_sum_to_one_all_ntas(db_conn, tmp_path: Path) -> None:  # type: ignore[misc]
    """For every 2010 source NTA in the crosswalk, weights sum to 1.0."""
    crosswalk_path = Path(os.environ.get("NTA_CROSSWALK_PATH", "data/nta_2010_2020_crosswalk.csv"))
    if not crosswalk_path.exists():
        crosswalk_path = crosswalk_path.with_suffix(".xlsx")
    if not crosswalk_path.exists():
        pytest.skip(f"Crosswalk file not found at {crosswalk_path}")

    if crosswalk_path.suffix == ".xlsx":
        df = pd.read_excel(crosswalk_path)
        df.columns = df.columns.str.lower().str.strip()
        tmp_csv = tmp_path / "crosswalk.csv"
        df.to_csv(tmp_csv, index=False)
        load_crosswalk(tmp_csv)
    else:
        load_crosswalk(crosswalk_path)

    from rat_ml.data.tract_crosswalk import all_nta2010_codes  # noqa: PLC0415

    bad: list[str] = []
    for nta2010 in all_nta2010_codes():
        w = weights_for(nta2010)
        total = sum(w.values())
        if abs(total - 1.0) > 1e-6:
            bad.append(f"{nta2010}: sum={total:.8f}")

    assert not bad, f"Weights do not sum to 1.0 for {len(bad)} source NTAs:\n" + "\n".join(bad)


@pytest.mark.integration
async def test_nta_boundaries_within_nyc_bbox(db_conn) -> None:  # type: ignore[misc]
    """All geometries in raw.nta_boundaries must fall within the NYC WGS84 bounding box.

    Catches EPSG:2263 (NY State Plane, coordinates ~900k–1.1M ft) accidentally
    stored instead of EPSG:4326 (WGS84, lon/lat).  Any projection bug in
    load_nta_boundaries.py or a future ingest script will fail this test.

    Bounding box: minx=-74.30, miny=40.40, maxx=-73.65, maxy=40.95
    """
    rows = await db_conn.fetch(
        """
        SELECT nta_id,
               ST_XMin(geom) AS xmin,
               ST_XMax(geom) AS xmax,
               ST_YMin(geom) AS ymin,
               ST_YMax(geom) AS ymax
        FROM raw.nta_boundaries
        WHERE geom IS NOT NULL
          AND NOT (
              ST_XMin(geom) >= $1 AND ST_XMax(geom) <= $2
              AND ST_YMin(geom) >= $3 AND ST_YMax(geom) <= $4
          )
        """,
        NYC_MINX, NYC_MAXX, NYC_MINY, NYC_MAXY,
    )
    assert not rows, (
        f"{len(rows)} NTA geometries fall outside the NYC bounding box "
        f"({NYC_MINX},{NYC_MINY} → {NYC_MAXX},{NYC_MAXY}). "
        f"Likely cause: coordinates stored in NY State Plane (EPSG:2263) "
        f"instead of WGS84 (EPSG:4326). Offending NTAs: "
        f"{[r['nta_id'] for r in rows]}"
    )
