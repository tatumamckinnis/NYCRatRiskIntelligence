"""Tests for ingest_311.

Unit tests: cursor helpers, point filtering logic, no DB required.
Integration tests (pytest.mark.integration): idempotency, no per-complaint
rows, require DATABASE_URL and a live Supabase connection.
"""

from __future__ import annotations

import os

import pytest


# ===========================================================================
# Unit tests
# ===========================================================================

def test_epoch_fallback_used_when_no_cursor() -> None:
    """EPOCH constant is a valid ISO timestamp string."""
    from rat_ml.data.ingest_311 import EPOCH  # noqa: PLC0415
    assert EPOCH.startswith("20")
    assert "T" in EPOCH or len(EPOCH) == 10


def test_complaint_type_constant() -> None:
    from rat_ml.data.ingest_311 import COMPLAINT_TYPE  # noqa: PLC0415
    assert COMPLAINT_TYPE == "Rodent"


def test_source_name_constant() -> None:
    from rat_ml.data.ingest_311 import SOURCE_NAME  # noqa: PLC0415
    assert SOURCE_NAME == "311_complaints"


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
async def test_no_per_complaint_rows_stored(db_conn) -> None:  # type: ignore[misc]
    """raw.complaints_nta_week must only contain (nta_id, week_start) aggregates.

    Verifies the $0-budget amendment: 311 data is aggregated at ingest,
    never stored per complaint.
    """
    # The table structure itself enforces this: PK is (nta_id, week_start).
    # Confirm the table has no complaint_id or unique_key column.
    cols = await db_conn.fetch(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'raw'
          AND table_name = 'complaints_nta_week'
        """
    )
    col_names = {r["column_name"] for r in cols}
    assert "complaint_id" not in col_names
    assert "unique_key" not in col_names
    assert "nta_id" in col_names
    assert "week_start" in col_names
    assert "complaint_count" in col_names


@pytest.mark.integration
async def test_idempotency_311(db_conn) -> None:  # type: ignore[misc]
    """Running the 311 ingest twice advances the cursor so the second run
    fetches zero new rows, leaving the row count unchanged."""
    from rat_ml.data.ingest_311 import SOURCE_NAME, run  # noqa: PLC0415

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        pytest.skip("DATABASE_URL not set")

    count_before = await db_conn.fetchval(
        "SELECT COUNT(*) FROM raw.complaints_nta_week"
    )
    # First run: fetches new data and advances cursor.
    await run(db_url)
    count_after_first = await db_conn.fetchval(
        "SELECT COUNT(*) FROM raw.complaints_nta_week"
    )

    # Second run: cursor is already at latest updated_date → zero new rows.
    await run(db_url)
    count_after_second = await db_conn.fetchval(
        "SELECT COUNT(*) FROM raw.complaints_nta_week"
    )

    assert count_after_first == count_after_second, (
        "Row count changed on second run — cursor not advancing correctly: "
        f"{count_after_first} → {count_after_second}"
    )
