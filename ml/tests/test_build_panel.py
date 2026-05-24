"""Tests for panel assembly (T-08).

Unit tests: SQL generation helpers, no DB required.
Integration tests (pytest.mark.integration): upsert idempotency and
column coverage, require DATABASE_URL and a live Supabase connection.
"""

from __future__ import annotations

import os

import pytest

from rat_ml.features.panel import build_panel_sql
from rat_ml.features.temporal import COMMERCIAL_LANDUSE, RESIDENTIAL_LANDUSE


# ===========================================================================
# Unit tests — SQL generation
# ===========================================================================

def test_build_panel_sql_returns_string() -> None:
    sql = build_panel_sql()
    assert isinstance(sql, str)
    assert len(sql) > 100


def test_build_panel_sql_contains_insert() -> None:
    assert "INSERT INTO features.nta_week_panel" in build_panel_sql()


def test_build_panel_sql_uses_st_within() -> None:
    """Labels step must use PostGIS spatial join (Option A)."""
    assert "ST_Within" in build_panel_sql()


def test_build_panel_sql_references_all_raw_tables() -> None:
    sql = build_panel_sql()
    for table in (
        "raw.rodent_inspections",
        "raw.nta_boundaries",
        "raw.complaints_nta_week",
        "raw.restaurant_inspections",
        "raw.dob_permits",
        "raw.weather_daily",
        "raw.pluto",
    ):
        assert table in sql, f"Expected {table!r} in build_panel_sql()"


def test_build_panel_sql_uses_residential_landuse_codes() -> None:
    sql = build_panel_sql()
    for code in RESIDENTIAL_LANDUSE:
        assert f"'{code}'" in sql


def test_build_panel_sql_uses_commercial_landuse_codes() -> None:
    sql = build_panel_sql()
    for code in COMMERCIAL_LANDUSE:
        assert f"'{code}'" in sql


def test_build_panel_sql_has_on_conflict() -> None:
    """Upsert must be idempotent."""
    assert "ON CONFLICT" in build_panel_sql()


def test_build_panel_sql_has_lag_window_functions() -> None:
    sql = build_panel_sql()
    assert "LAG(complaint_count," in sql
    assert "PARTITION BY nta_id" in sql


def test_build_panel_sql_has_weather_aggregates() -> None:
    sql = build_panel_sql()
    assert "AVG(tavg_c)" in sql
    assert "SUM(prcp_mm)" in sql
    assert "SUM(hdd)" in sql
    assert "SUM(cdd)" in sql


def test_build_panel_sql_has_percentile_for_year_built() -> None:
    sql = build_panel_sql()
    assert "PERCENTILE_CONT" in sql
    assert "yearbuilt" in sql


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
async def test_panel_upsert_idempotent(db_conn) -> None:  # type: ignore[misc]
    """Running build_panel twice must not change the row count."""
    from rat_ml.features.panel import run  # noqa: PLC0415

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        pytest.skip("DATABASE_URL not set")

    await run(db_url)
    count_after_first = await db_conn.fetchval(
        "SELECT COUNT(*) FROM features.nta_week_panel"
    )

    await run(db_url)
    count_after_second = await db_conn.fetchval(
        "SELECT COUNT(*) FROM features.nta_week_panel"
    )

    assert count_after_first == count_after_second, (
        f"Row count changed on second run: {count_after_first} → {count_after_second}"
    )


@pytest.mark.integration
async def test_panel_has_expected_columns(db_conn) -> None:  # type: ignore[misc]
    """features.nta_week_panel must contain all T-08 columns."""
    rows = await db_conn.fetch(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'features'
          AND table_name   = 'nta_week_panel'
        """
    )
    cols = {r["column_name"] for r in rows}
    required = {
        "nta_id", "week_start",
        "active_rat_signs_count", "inspections_count",
        "active_rat_signs_rate", "active_rat_signs_ind",
        "complaints_count", "complaints_lag_1w", "complaints_lag_4w", "complaints_lag_12w",
        "rest_pest_violations_count",
        "permits_active_count", "demolitions_count",
        "weather_tavg_c", "weather_prcp_mm", "weather_hdd", "weather_cdd",
        "units_total", "year_built_median",
        "landuse_residential_pct", "landuse_commercial_pct",
    }
    missing = required - cols
    assert not missing, f"Missing columns in nta_week_panel: {missing}"


@pytest.mark.integration
async def test_panel_rate_in_range(db_conn) -> None:  # type: ignore[misc]
    """active_rat_signs_rate must be between 0 and 1 (inclusive) where non-NULL."""
    rows = await db_conn.fetch(
        """
        SELECT nta_id, week_start, active_rat_signs_rate
        FROM features.nta_week_panel
        WHERE active_rat_signs_rate IS NOT NULL
          AND (active_rat_signs_rate < 0 OR active_rat_signs_rate > 1)
        LIMIT 10
        """
    )
    assert not rows, (
        f"{len(rows)} rows have active_rat_signs_rate outside [0,1]. "
        f"Sample: {[dict(r) for r in rows[:3]]}"
    )
