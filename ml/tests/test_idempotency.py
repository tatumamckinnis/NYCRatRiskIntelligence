"""T-13: Combined idempotency tests for all Phase 1 ingest scripts.

Each test runs an ingest script's run() function twice and asserts that
the row count in the target table is identical after both runs.  This
verifies that all upsert logic (ON CONFLICT DO UPDATE / DO NOTHING) is
correct and that no duplicate rows are inserted on re-run.

All tests are integration tests (pytest.mark.integration) that require
DATABASE_URL and valid Socrata/Meteostat connectivity (or pre-loaded data).

Scripts covered:
    - ingest_rodent_inspections  → raw.rodent_inspections
    - ingest_311                 → raw.complaints_nta_week
    - ingest_restaurant_inspections → raw.restaurant_inspections
    - ingest_dob_permits         → raw.dob_permits
    - ingest_weather             → raw.weather_daily

Note: ingest_pluto is excluded because it requires a download URL argument
and a ~860k-row bulk load that would be impractical to run twice in CI.
Its idempotency is enforced by the ON CONFLICT DO UPDATE in its own
upsert SQL.
"""

from __future__ import annotations

import os

import pytest


def _db_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    return url


async def _count(table: str, db_url: str) -> int:
    import asyncpg  # noqa: PLC0415
    conn = await asyncpg.connect(db_url)
    try:
        return int(await conn.fetchval(f"SELECT COUNT(*) FROM {table}"))  # noqa: S608
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Rodent inspections
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_idempotency_rodent_inspections() -> None:
    """Running ingest_rodent_inspections twice must not change the row count."""
    from rat_ml.data.ingest_rodent_inspections import run  # noqa: PLC0415

    db_url = _db_url()
    await run(db_url)
    count_1 = await _count("raw.rodent_inspections", db_url)

    await run(db_url)
    count_2 = await _count("raw.rodent_inspections", db_url)

    assert count_1 == count_2, (
        f"raw.rodent_inspections row count changed on second run: {count_1} → {count_2}"
    )


# ---------------------------------------------------------------------------
# 311 complaints
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_idempotency_311() -> None:
    """Running ingest_311 twice must not change the row count in raw.complaints_nta_week."""
    from rat_ml.data.ingest_311 import run  # noqa: PLC0415

    db_url = _db_url()
    await run(db_url)
    count_1 = await _count("raw.complaints_nta_week", db_url)

    await run(db_url)
    count_2 = await _count("raw.complaints_nta_week", db_url)

    assert count_1 == count_2, (
        f"raw.complaints_nta_week row count changed on second run: {count_1} → {count_2}"
    )


# ---------------------------------------------------------------------------
# Restaurant inspections
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_idempotency_restaurant_inspections() -> None:
    """Running ingest_restaurant_inspections twice must not change the row count."""
    from rat_ml.data.ingest_restaurant_inspections import run  # noqa: PLC0415

    db_url = _db_url()
    await run(db_url)
    count_1 = await _count("raw.restaurant_inspections", db_url)

    await run(db_url)
    count_2 = await _count("raw.restaurant_inspections", db_url)

    assert count_1 == count_2, (
        f"raw.restaurant_inspections row count changed on second run: {count_1} → {count_2}"
    )


# ---------------------------------------------------------------------------
# DOB permits
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_idempotency_dob_permits() -> None:
    """Running ingest_dob_permits twice must not change the row count."""
    from rat_ml.data.ingest_dob_permits import run  # noqa: PLC0415

    db_url = _db_url()
    await run(db_url)
    count_1 = await _count("raw.dob_permits", db_url)

    await run(db_url)
    count_2 = await _count("raw.dob_permits", db_url)

    assert count_1 == count_2, (
        f"raw.dob_permits row count changed on second run: {count_1} → {count_2}"
    )


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_idempotency_weather() -> None:
    """Running ingest_weather twice must not change the row count in raw.weather_daily."""
    from rat_ml.data.ingest_weather import run  # noqa: PLC0415

    db_url = _db_url()
    await run(db_url)
    count_1 = await _count("raw.weather_daily", db_url)

    await run(db_url)
    count_2 = await _count("raw.weather_daily", db_url)

    assert count_1 == count_2, (
        f"raw.weather_daily row count changed on second run: {count_1} → {count_2}"
    )


# ---------------------------------------------------------------------------
# Panel assembly (build_panel is also idempotent via ON CONFLICT)
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_idempotency_panel() -> None:
    """Running build_panel twice must not change the row count in features.nta_week_panel."""
    from rat_ml.features.panel import run  # noqa: PLC0415

    db_url = _db_url()
    await run(db_url)
    count_1 = await _count("features.nta_week_panel", db_url)

    await run(db_url)
    count_2 = await _count("features.nta_week_panel", db_url)

    assert count_1 == count_2, (
        f"features.nta_week_panel row count changed on second run: {count_1} → {count_2}"
    )
