"""Ingest DOB permit records into raw.dob_permits.

Sources:
  ipu4-2q9a — DOB NOW: Build – Approved Permits
  rbx6-tga4 — DOB Permit Issuance (legacy)

Strategy: Incremental on issuance_date; cursor stored in raw.ingest_cursors
          separately per source ('dob_permits_now' and 'dob_permits_legacy').

Usage (from repo root):
    uv run --package rat-ml python ml/src/rat_ml/data/ingest_dob_permits.py
"""

from __future__ import annotations

import asyncio
import os
import sys

import asyncpg

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

from rat_ml.data._socrata import get_client, paginate
from rat_ml.data.bbl_join import emit_unmatched_report, normalize_bbl

DATASETS: list[tuple[str, str]] = [
    ("ipu4-2q9a", "dob_permits_now"),
    ("rbx6-tga4", "dob_permits_legacy"),
]
EPOCH = "1990-01-01"


async def _get_cursor(conn: asyncpg.Connection, source: str) -> str:
    row = await conn.fetchrow(
        "SELECT last_cursor_value FROM raw.ingest_cursors WHERE source = $1", source
    )
    return row["last_cursor_value"] if row and row["last_cursor_value"] else EPOCH


async def _set_cursor(
    conn: asyncpg.Connection, source: str, cursor: str, rows_added: int
) -> None:
    await conn.execute(
        """
        INSERT INTO raw.ingest_cursors (source, last_cursor_value, last_run_at, rows_ingested_total)
        VALUES ($1, $2, NOW(), $3)
        ON CONFLICT (source) DO UPDATE SET
            last_cursor_value   = EXCLUDED.last_cursor_value,
            last_run_at         = EXCLUDED.last_run_at,
            rows_ingested_total = raw.ingest_cursors.rows_ingested_total + EXCLUDED.rows_ingested_total
        """,
        source, cursor, rows_added,
    )


def _parse_row(row: dict, source: str) -> dict | None:
    raw_bbl = row.get("bbl") or row.get("bin__s_bbl")
    bbl = normalize_bbl(raw_bbl)

    # Both datasets expose slightly different field names.
    issuance = (
        row.get("issuance_date")
        or row.get("issued_date")
        or row.get("filing_date")
    )
    if not issuance:
        return None

    # Construct a stable permit key from source + a dataset-specific identifier.
    permit_num = (
        row.get("job__")
        or row.get("job_filing_number")
        or row.get("initial_cost")  # last-resort
    )
    permit_key = f"{source}_{permit_num}_{issuance[:10]}" if permit_num else None
    if not permit_key:
        return None

    return {
        "permit_key": permit_key,
        "bbl": bbl,
        "raw_bbl_missing": bool(raw_bbl and bbl is None),
        "bin": row.get("bin") or row.get("bin_"),
        "borough": row.get("borough") or row.get("boro"),
        "issuance_date": issuance[:10],
        "expiration_date": (row.get("expiration_date") or "")[:10] or None,
        "job_type": row.get("job_type") or row.get("job__type"),
        "work_type": row.get("work_type"),
        "source": "now" if "now" in source else "legacy",
    }


async def upsert_batch(
    conn: asyncpg.Connection, rows: list[dict], source: str
) -> tuple[int, int]:
    parsed = [_parse_row(r, source) for r in rows]
    valid = [p for p in parsed if p is not None]
    unmatched = sum(1 for p in valid if p["raw_bbl_missing"])

    async with conn.transaction():
        for p in valid:
            await conn.execute(
                """
                INSERT INTO raw.dob_permits (
                    permit_key, bbl, bin, borough,
                    issuance_date, expiration_date, job_type, work_type, source
                ) VALUES ($1, $2, $3, $4, $5::date, $6::date, $7, $8, $9)
                ON CONFLICT (permit_key) DO UPDATE SET
                    bbl             = EXCLUDED.bbl,
                    expiration_date = EXCLUDED.expiration_date,
                    job_type        = EXCLUDED.job_type,
                    work_type       = EXCLUDED.work_type
                """,
                p["permit_key"], p["bbl"], p["bin"], p["borough"],
                p["issuance_date"], p["expiration_date"],
                p["job_type"], p["work_type"], p["source"],
            )

    return len(valid), unmatched


async def ingest_dataset(
    conn: asyncpg.Connection, dataset_id: str, source: str
) -> None:
    cursor = await _get_cursor(conn, source)
    print(f"[{source}] fetching since {cursor}")
    client = get_client()
    total_rows = total_unmatched = 0
    latest = cursor

    for batch, offset in paginate(
        client,
        dataset_id,
        where=f"issuance_date >= '{cursor}'",
        order="issuance_date ASC",
    ):
        n, u = await upsert_batch(conn, batch, source)
        total_rows += n
        total_unmatched += u

        for row in batch:
            d = row.get("issuance_date", "")
            if d > latest:
                latest = d

        print(f"  [{source}] offset={offset:>7}  upserted={n}  unmatched_bbl={u}")

    await _set_cursor(conn, source, latest, total_rows)
    emit_unmatched_report(source, total=total_rows, unmatched=total_unmatched)
    print(f"[{source}] Done. total={total_rows}  unmatched_bbl={total_unmatched}")


async def run(db_url: str) -> None:
    conn = await asyncpg.connect(db_url)
    for dataset_id, source in DATASETS:
        await ingest_dataset(conn, dataset_id, source)
    await conn.close()


async def main() -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DATABASE_URL is not set.")
    await run(db_url)


if __name__ == "__main__":
    asyncio.run(main())
