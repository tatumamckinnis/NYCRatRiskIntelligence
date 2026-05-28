"""Ingest NYC MapPLUTO lot data into raw.pluto.

Source:  NYC DCP MapPLUTO (latest vintage) — downloaded as a CSV zip.
         ~860k rows. Run once per quarter; subsequent runs are full upserts.
Key:     bbl (10-char normalized)

The default URL points to the latest PLUTO on NYC Open Data. Override with
--url if DCP releases a new vintage (the dataset ID changes each quarter).

Usage (from repo root):
    uv run --package rat-ml python ml/src/rat_ml/data/ingest_pluto.py
    uv run --package rat-ml python ml/src/rat_ml/data/ingest_pluto.py --url <url>
"""

from __future__ import annotations

import argparse
import asyncio
import io
import os
import sys
import zipfile
from pathlib import Path

import asyncpg
import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

from rat_ml.data.bbl_join import emit_unmatched_report, normalize_bbl, resolve_bbl

# NYC Open Data — MapPLUTO (latest vintage, 25v4+).
# Verify/update at: https://data.cityofnewyork.us/City-Government/Primary-Land-Use-Tax-Lot-Output-PLUTO-/64uk-42ks
DEFAULT_URL = (
    "https://data.cityofnewyork.us/api/views/64uk-42ks/rows.csv?accessType=DOWNLOAD"
)
SOURCE_NAME = "pluto"
BATCH_SIZE = 5_000


def _download_csv(url: str) -> pd.DataFrame:
    print(f"Downloading PLUTO from {url} …")
    resp = requests.get(url, timeout=300, stream=True)
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "")
    raw = resp.content

    if "zip" in content_type or url.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, dtype=str, low_memory=False)
    else:
        df = pd.read_csv(io.BytesIO(raw), dtype=str, low_memory=False)

    df.columns = df.columns.str.lower().str.strip()
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def _parse_df(df: pd.DataFrame) -> tuple[list[dict], int]:
    """Normalise PLUTO columns to the raw.pluto schema."""
    col = {c: c for c in df.columns}

    def pick(*candidates: str) -> str | None:
        for c in candidates:
            if c in col:
                return c
        return None

    bbl_col = pick("bbl", "parid")
    appbbl_col = pick("appbbl")
    nta_col = pick("nta2020", "nta", "ntacode")

    rows = []
    unmatched = 0

    for _, r in df.iterrows():
        raw_bbl = r.get(bbl_col) if bbl_col else None
        raw_appbbl = r.get(appbbl_col) if appbbl_col else None

        bbl = normalize_bbl(raw_bbl)
        appbbl = normalize_bbl(raw_appbbl)

        if bbl is None:
            if raw_bbl:
                unmatched += 1
            continue

        effective_bbl = resolve_bbl(bbl, appbbl)

        def _int(col_name: str) -> int | None:
            v = r.get(col_name)
            try:
                return int(float(v)) if v and str(v).strip() else None
            except (ValueError, TypeError):
                return None

        def _float(col_name: str) -> float | None:
            v = r.get(col_name)
            try:
                return float(v) if v and str(v).strip() else None
            except (ValueError, TypeError):
                return None

        rows.append({
            "bbl": effective_bbl,
            "unitsres": _int("unitsres"),
            "unitstotal": _int("unitstotal"),
            "yearbuilt": _int("yearbuilt"),
            "landuse": str(r.get("landuse", "") or "").strip()[:2] or None,
            "bldgclass": str(r.get("bldgclass", "") or "").strip() or None,
            "appbbl": appbbl,
            "nta2020": str(r.get(nta_col, "") or "").strip() or None if nta_col else None,
            "latitude": _float("latitude"),
            "longitude": _float("longitude"),
        })

    return rows, unmatched


_SQL = """
    INSERT INTO raw.pluto (
        bbl, unitsres, unitstotal, yearbuilt,
        landuse, bldgclass, appbbl, nta2020, latitude, longitude
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
    ON CONFLICT (bbl) DO UPDATE SET
        unitsres   = EXCLUDED.unitsres,
        unitstotal = EXCLUDED.unitstotal,
        yearbuilt  = EXCLUDED.yearbuilt,
        landuse    = EXCLUDED.landuse,
        bldgclass  = EXCLUDED.bldgclass,
        appbbl     = EXCLUDED.appbbl,
        nta2020    = EXCLUDED.nta2020,
        latitude   = EXCLUDED.latitude,
        longitude  = EXCLUDED.longitude,
        ingested_at= NOW()
"""


async def upsert_rows(conn: asyncpg.Connection, rows: list[dict]) -> None:
    args = [
        (p["bbl"], p["unitsres"], p["unitstotal"], p["yearbuilt"],
         p["landuse"], p["bldgclass"], p["appbbl"], p["nta2020"],
         p["latitude"], p["longitude"])
        for p in rows
    ]
    await conn.executemany(_SQL, args)


async def run(db_url: str, url: str) -> None:
    df = _download_csv(url)
    rows, unmatched = _parse_df(df)
    print(f"  Parsed {len(rows):,} valid rows, {unmatched} unmatched BBL")

    conn = await asyncpg.connect(db_url)
    await conn.execute("SET statement_timeout = 0")
    total = 0
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        await upsert_rows(conn, batch)
        total += len(batch)
        print(f"  upserted {total:,} / {len(rows):,}")

    # Populate nta2020 via PostGIS spatial join against NTA 2020 boundaries.
    print("Populating nta2020 via spatial join …")
    result = await conn.execute("""
        UPDATE raw.pluto p
        SET nta2020 = nb.nta_id
        FROM raw.nta_boundaries nb
        WHERE p.nta2020 IS NULL
          AND p.latitude IS NOT NULL
          AND p.longitude IS NOT NULL
          AND ST_Within(
              ST_SetSRID(ST_Point(p.longitude, p.latitude), 4326),
              nb.geom
          )
    """)
    print(f"  nta2020 set for {result.split()[-1]} lots")

    emit_unmatched_report(SOURCE_NAME, total=len(rows) + unmatched, unmatched=unmatched)
    print(f"Done. upserted={total:,}  unmatched_bbl={unmatched}")
    await conn.close()


async def main() -> None:
    db_url = os.environ.get("DIRECT_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DIRECT_DATABASE_URL or DATABASE_URL is not set.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="PLUTO CSV download URL")
    args = parser.parse_args()
    await run(db_url, args.url)


if __name__ == "__main__":
    asyncio.run(main())
