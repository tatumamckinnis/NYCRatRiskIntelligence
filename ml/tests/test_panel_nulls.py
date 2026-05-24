"""T-13: NULL-rate tests for features.nta_week_panel required columns.

Per SPEC §10.1, the following columns must never be NULL in the panel:
    - nta_id
    - week_start
    - active_rat_signs_count
    - inspections_count
    - active_rat_signs_ind

This file also verifies the Phase 1 acceptance criterion:
    features.nta_week_panel has > 100,000 rows.

All tests are integration tests (pytest.mark.integration) that require
DATABASE_URL and a fully assembled panel (build_panel.py completed).
"""

from __future__ import annotations

import os

import pytest

# Required columns that must have zero NULLs (SPEC §10.1)
REQUIRED_NONNULL = [
    "nta_id",
    "week_start",
    "active_rat_signs_count",
    "inspections_count",
    "active_rat_signs_ind",
]

# Phase 1 acceptance threshold
MIN_PANEL_ROWS = 100_000


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
async def test_panel_meets_row_count_acceptance_criterion(db_conn) -> None:
    """features.nta_week_panel must contain > 100,000 rows (SPEC acceptance check)."""
    n = await db_conn.fetchval("SELECT COUNT(*) FROM features.nta_week_panel")
    assert n and int(n) > MIN_PANEL_ROWS, (
        f"Panel has {n:,} rows; need > {MIN_PANEL_ROWS:,}. "
        "Run build_panel.py after all ingest scripts have loaded data."
    )


@pytest.mark.integration
async def test_panel_required_columns_have_no_nulls(db_conn) -> None:
    """No required column may contain a NULL value anywhere in the panel."""
    null_checks = ",\n    ".join(
        f"COUNT(*) FILTER (WHERE {col} IS NULL) AS {col}"
        for col in REQUIRED_NONNULL
    )
    sql = f"SELECT {null_checks} FROM features.nta_week_panel"  # noqa: S608
    row = await db_conn.fetchrow(sql)
    if row is None:
        pytest.skip("Panel is empty")

    violations = {col: int(row[col]) for col in REQUIRED_NONNULL if row[col] and int(row[col]) > 0}
    assert not violations, (
        f"Required columns contain NULLs: {violations}. "
        "Run build_panel.py to repair."
    )


@pytest.mark.integration
async def test_panel_nta_id_never_null(db_conn) -> None:
    """nta_id must never be NULL — it is half of the primary key."""
    n = await db_conn.fetchval(
        "SELECT COUNT(*) FROM features.nta_week_panel WHERE nta_id IS NULL"
    )
    assert int(n) == 0, f"{n} rows have NULL nta_id"


@pytest.mark.integration
async def test_panel_week_start_never_null(db_conn) -> None:
    """week_start must never be NULL — it is half of the primary key."""
    n = await db_conn.fetchval(
        "SELECT COUNT(*) FROM features.nta_week_panel WHERE week_start IS NULL"
    )
    assert int(n) == 0, f"{n} rows have NULL week_start"


@pytest.mark.integration
async def test_panel_active_rat_signs_count_never_null(db_conn) -> None:
    n = await db_conn.fetchval(
        "SELECT COUNT(*) FROM features.nta_week_panel WHERE active_rat_signs_count IS NULL"
    )
    assert int(n) == 0, f"{n} rows have NULL active_rat_signs_count"


@pytest.mark.integration
async def test_panel_inspections_count_never_null(db_conn) -> None:
    n = await db_conn.fetchval(
        "SELECT COUNT(*) FROM features.nta_week_panel WHERE inspections_count IS NULL"
    )
    assert int(n) == 0, f"{n} rows have NULL inspections_count"


@pytest.mark.integration
async def test_panel_active_rat_signs_ind_never_null(db_conn) -> None:
    n = await db_conn.fetchval(
        "SELECT COUNT(*) FROM features.nta_week_panel WHERE active_rat_signs_ind IS NULL"
    )
    assert int(n) == 0, f"{n} rows have NULL active_rat_signs_ind"


@pytest.mark.integration
async def test_panel_covers_all_ntas(db_conn) -> None:
    """Panel must contain rows for all 262 NTAs in raw.nta_boundaries."""
    panel_ntas = await db_conn.fetchval(
        "SELECT COUNT(DISTINCT nta_id) FROM features.nta_week_panel"
    )
    boundary_ntas = await db_conn.fetchval(
        "SELECT COUNT(*) FROM raw.nta_boundaries"
    )
    if not boundary_ntas:
        pytest.skip("raw.nta_boundaries is empty")
    assert int(panel_ntas) == int(boundary_ntas), (
        f"Panel covers {panel_ntas} NTAs but raw.nta_boundaries has {boundary_ntas}. "
        "Some NTAs have no inspection history or failed the spatial join."
    )
