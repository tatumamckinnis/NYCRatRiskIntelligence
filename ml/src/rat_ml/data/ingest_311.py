"""Ingest 311 rodent complaints (NTA-week aggregation) into raw.complaints_nta_week.

Source:  NYC Open Data — 311 Service Requests (Socrata erm2-nwe9)
Filter:  complaint_type = 'Rodent'
Strategy: Incremental — cursor on updated_date stored in raw.ingest_cursors.
          New complaints are spatially joined to NTA 2020 boundaries via
          PostGIS ST_Within, then aggregated to (nta_id, week_start, count).
          No per-complaint rows are stored (per $0-budget amendment).

Usage (from repo root):
    uv run --package rat-ml python ml/src/rat_ml/data/ingest_311.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone

import asyncpg

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

from rat_ml.data._socrata import get_client, paginate

DATASET_ID = "erm2-nwe9"
SOURCE_NAME = "311_complaints"
COMPLAINT_TYPE = "Rodent"
# ISO timestamp used as the initial cursor when no prior run exists.
EPOCH = "2010-01-01T00:00:00.000"


async def _get_cursor(conn: asyncpg.Connection) -> str:
    row = await conn.fetchrow(
        "SELECT last_cursor_value FROM raw.ingest_cursors WHERE source = $1",
        SOURCE_NAME,
    )
    return row["last_cursor_value"] if row and row["last_cursor_value"] else EPOCH


async def _set_cursor(conn: asyncpg.Connection, cursor: str, rows_added: int) -> None:
    await conn.execute(
        """
        INSERT INTO raw.ingest_cursors (source, last_cursor_value, last_run_at, rows_ingested_total)
        VALUES ($1, $2, NOW(), $3)
        ON CONFLICT (source) DO UPDATE SET
            last_cursor_value   = EXCLUDED.last_cursor_value,
            last_run_at         = EXCLUDED.last_run_at,
            rows_ingested_total = raw.ingest_cursors.rows_ingested_total + EXCLUDED.rows_ingested_total
        """,
        SOURCE_NAME,
        cursor,
        rows_added,
    )


async def _spatial_aggregate_and_upsert(
    conn: asyncpg.Connection,
    complaints: list[dict],
) -> int:
    """Spatially join complaints to NTA boundaries and upsert NTA-week aggregates.

    Uses a PostgreSQL VALUES clause to avoid a round-trip temp table.
    Returns the number of (nta_id, week_start) pairs upserted.
    """
    # Filter rows that have usable lat/lon.
    points = []
    for row in complaints:
        lat = row.get("latitude")
        lon = row.get("longitude")
        created = row.get("created_date")
        if lat and lon and created:
            try:
                points.append((float(lon), float(lat), created[:10]))  # date portion only
            except (ValueError, TypeError):
                pass

    if not points:
        return 0

    # Build a VALUES list and do the spatial join + aggregation in one SQL statement.
    # Each point: (lon, lat, created_date_str)
    values_sql = ", ".join(
        f"(ST_SetSRID(ST_Point({lon}, {lat}), 4326), '{d}'::date)"
        for lon, lat, d in points
    )

    result = await conn.execute(
        f"""
        WITH pts AS (
            SELECT geom, created_date
            FROM (VALUES {values_sql}) AS t(geom, created_date)
        ),
        joined AS (
            SELECT
                nb.nta_id,
                date_trunc('week', pts.created_date)::date AS week_start
            FROM pts
            JOIN raw.nta_boundaries nb ON ST_Within(pts.geom, nb.geom)
        ),
        agg AS (
            SELECT nta_id, week_start, COUNT(*)::integer AS cnt
            FROM joined
            GROUP BY nta_id, week_start
        )
        INSERT INTO raw.complaints_nta_week (nta_id, week_start, complaint_count)
        SELECT nta_id, week_start, cnt FROM agg
        ON CONFLICT (nta_id, week_start) DO UPDATE SET
            complaint_count = raw.complaints_nta_week.complaint_count + EXCLUDED.complaint_count
        """,
    )
    # asyncpg execute returns e.g. "INSERT 0 5"
    try:
        return int(result.split()[-1])
    except (IndexError, ValueError):
        return 0


async def run(db_url: str) -> None:
    conn = await asyncpg.connect(db_url)
    await conn.execute("SET statement_timeout = 0")
    cursor = await _get_cursor(conn)
    print(f"Fetching 311 rodent complaints created after {cursor}")

    client = get_client()
    total_complaints = 0
    total_nta_weeks = 0
    latest_updated = cursor

    for batch, offset in paginate(
        client,
        DATASET_ID,
        where=(
            f"complaint_type='{COMPLAINT_TYPE}' "
            f"AND created_date > '{cursor}'"
        ),
        order="created_date ASC",
        select="unique_key,created_date,latitude,longitude",
    ):
        n = await _spatial_aggregate_and_upsert(conn, batch)
        total_nta_weeks += n
        total_complaints += len(batch)

        # Track the latest created_date seen in this batch.
        for row in batch:
            ud = row.get("created_date", "")
            if ud > latest_updated:
                latest_updated = ud

        print(f"  offset={offset:>7}  complaints={len(batch)}  nta_week_pairs={n}")

    await _set_cursor(conn, latest_updated, total_complaints)
    print(
        f"Done. complaints_fetched={total_complaints}  "
        f"nta_week_upserts={total_nta_weeks}  "
        f"new_cursor={latest_updated}"
    )
    await conn.close()


async def main() -> None:
    db_url = os.environ.get("DIRECT_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DIRECT_DATABASE_URL or DATABASE_URL is not set.")
    await run(db_url)


if __name__ == "__main__":
    asyncio.run(main())
