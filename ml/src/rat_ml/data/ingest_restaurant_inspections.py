"""Ingest DOHMH restaurant inspection records into raw.restaurant_inspections.

Source:  NYC Open Data — DOHMH New York City Restaurant Inspection Results
         (Socrata 43nn-pn8j)
Key:     record_id = camis || '_' || inspection_date || '_' || violation_code
Pest:    Violation codes 04K (live roaches), 04L (evidence of mice/rats),
         08A (facility not free from pests) are flagged is_pest_violation=TRUE.

Usage (from repo root):
    uv run --package rat-ml python ml/src/rat_ml/data/ingest_restaurant_inspections.py
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

DATASET_ID = "43nn-pn8j"
SOURCE_NAME = "restaurant_inspections"
PEST_CODES = frozenset({"04K", "04L", "08A"})


def _parse_row(row: dict) -> dict | None:
    camis = row.get("camis")
    insp_date = row.get("inspection_date", "")
    violation_code = (row.get("violation_code") or "").strip().upper()

    if not camis or not insp_date:
        return None

    record_id = f"{camis}_{insp_date[:10]}_{violation_code}"
    raw_bbl = row.get("bbl")
    bbl = normalize_bbl(raw_bbl)

    try:
        score = int(row["score"]) if row.get("score") else None
    except (ValueError, TypeError):
        score = None

    return {
        "record_id": record_id,
        "camis": str(camis),
        "bbl": bbl,
        "raw_bbl_missing": bool(raw_bbl and bbl is None),
        "inspection_date": insp_date[:10],
        "violation_code": violation_code or None,
        "is_pest_violation": violation_code in PEST_CODES,
        "grade": row.get("grade"),
        "score": score,
    }


async def upsert_batch(conn: asyncpg.Connection, rows: list[dict]) -> tuple[int, int]:
    parsed = [_parse_row(r) for r in rows]
    valid = [p for p in parsed if p is not None]
    unmatched = sum(1 for p in valid if p["raw_bbl_missing"])

    async with conn.transaction():
        for p in valid:
            await conn.execute(
                """
                INSERT INTO raw.restaurant_inspections (
                    record_id, camis, bbl, inspection_date,
                    violation_code, is_pest_violation, grade, score
                ) VALUES ($1, $2, $3, $4::date, $5, $6, $7, $8)
                ON CONFLICT (record_id) DO UPDATE SET
                    bbl              = EXCLUDED.bbl,
                    is_pest_violation= EXCLUDED.is_pest_violation,
                    grade            = EXCLUDED.grade,
                    score            = EXCLUDED.score
                """,
                p["record_id"], p["camis"], p["bbl"],
                p["inspection_date"], p["violation_code"],
                p["is_pest_violation"], p["grade"], p["score"],
            )

    return len(valid), unmatched


async def run(db_url: str) -> None:
    conn = await asyncpg.connect(db_url)
    client = get_client()
    total_rows = total_unmatched = 0

    for batch, offset in paginate(client, DATASET_ID, order=":id"):
        n, u = await upsert_batch(conn, batch)
        total_rows += n
        total_unmatched += u
        print(f"  offset={offset:>7}  upserted={n}  unmatched_bbl={u}")

    emit_unmatched_report(SOURCE_NAME, total=total_rows, unmatched=total_unmatched)
    print(f"Done. total={total_rows}  unmatched_bbl={total_unmatched}")
    await conn.close()


async def main() -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DATABASE_URL is not set.")
    await run(db_url)


if __name__ == "__main__":
    asyncio.run(main())
