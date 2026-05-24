"""Tests for regime-indicator feature engineering (T-10).

Unit tests: SQL generation, no DB required.
Integration tests (pytest.mark.integration): spot-check specific rows in the
live panel to verify each regime flag is set correctly.
"""

from __future__ import annotations

import os

import pytest

from rat_ml.features.regime_indicators import build_regime_sql


# ===========================================================================
# Unit tests — SQL generation
# ===========================================================================

def test_regime_sql_is_update() -> None:
    assert build_regime_sql().strip().upper().startswith("UPDATE")


def test_regime_sql_targets_panel() -> None:
    assert "features.nta_week_panel" in build_regime_sql()


def test_regime_sql_sets_all_five_columns() -> None:
    sql = build_regime_sql()
    for col in (
        "regime_covid",
        "regime_8pm_setout",
        "regime_commercial_containerization",
        "regime_residential_containerization",
        "regime_rmz_active",
    ):
        assert col in sql, f"Expected column {col!r} in UPDATE SQL"


def test_regime_sql_rmz_is_false() -> None:
    sql = build_regime_sql()
    # regime_rmz_active must always be FALSE in Phase 1
    assert "regime_rmz_active" in sql
    after = sql[sql.index("regime_rmz_active"):]
    assert "FALSE" in after.split("\n")[0]


def test_regime_sql_covid_dates() -> None:
    sql = build_regime_sql()
    assert "2020-03-01" in sql
    assert "2020-06-30" in sql


def test_regime_sql_8pm_setout_date() -> None:
    assert "2023-04-01" in build_regime_sql()


def test_regime_sql_commercial_containerization_date() -> None:
    assert "2024-03-01" in build_regime_sql()


def test_regime_sql_residential_containerization_date() -> None:
    assert "2024-11-01" in build_regime_sql()


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
async def test_covid_row_flagged(db_conn) -> None:
    """A row from March 2020 must have regime_covid = TRUE and others = FALSE."""
    row = await db_conn.fetchrow(
        """
        SELECT regime_covid,
               regime_8pm_setout,
               regime_commercial_containerization,
               regime_residential_containerization,
               regime_rmz_active
        FROM features.nta_week_panel
        WHERE week_start = '2020-03-02'   -- Monday of the first COVID week
        LIMIT 1
        """
    )
    if row is None:
        pytest.skip("No panel rows for 2020-03-02 — run build_panel.py first")

    assert row["regime_covid"] is True
    assert row["regime_8pm_setout"] is False
    assert row["regime_commercial_containerization"] is False
    assert row["regime_residential_containerization"] is False
    assert row["regime_rmz_active"] is False


@pytest.mark.integration
async def test_november_2024_row_flagged(db_conn) -> None:
    """A row from November 2024 must have all four policy regimes TRUE."""
    row = await db_conn.fetchrow(
        """
        SELECT regime_covid,
               regime_8pm_setout,
               regime_commercial_containerization,
               regime_residential_containerization,
               regime_rmz_active
        FROM features.nta_week_panel
        WHERE week_start = '2024-11-04'   -- Monday in the first residential containerization week
        LIMIT 1
        """
    )
    if row is None:
        pytest.skip("No panel rows for 2024-11-04 — run build_panel.py first")

    assert row["regime_covid"] is False
    assert row["regime_8pm_setout"] is True
    assert row["regime_commercial_containerization"] is True
    assert row["regime_residential_containerization"] is True
    assert row["regime_rmz_active"] is False


@pytest.mark.integration
async def test_regime_run_updates_rows() -> None:
    """run() must report updating at least one row."""
    from rat_ml.features.regime_indicators import run  # noqa: PLC0415

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        pytest.skip("DATABASE_URL not set")

    n = await run(db_url)
    assert n > 0, "Expected run() to update at least one row in nta_week_panel"
