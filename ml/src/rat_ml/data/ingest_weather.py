"""Ingest daily weather data into raw.weather_daily.

Source:  Meteostat — station USW00094728 (Central Park, NYC)
Fetch:   Daily tavg/tmin/tmax/prcp/snow; compute HDD and CDD at ingest.
Key:     date (upsert)
Start:   2018-01-01 (covers the 3-year inspection window plus history)

Degree day base: 18°C (spec §5.3)
  HDD = max(18 - tavg, 0)  — heating demand proxy
  CDD = max(tavg - 18, 0)  — cooling demand proxy

Usage (from repo root):
    uv run --package rat-ml python ml/src/rat_ml/data/ingest_weather.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import date, datetime

import asyncpg

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

from meteostat import daily as Daily, Provider  # type: ignore[import-untyped]

# meteostat v2 station ID for Central Park / NYC Yorkville (40.7789, -73.9692).
# Use the station ID directly; Point-based lookup creates a synthetic station
# that has no provider data in v2.
STATION_ID = "KNYC0"
FETCH_START = datetime(2018, 1, 1)
DD_BASE = 18.0  # degree-day base in °C
SOURCE_NAME = "weather_daily"


def _compute_dd(tavg: float | None) -> tuple[float | None, float | None]:
    if tavg is None:
        return None, None
    hdd = max(DD_BASE - tavg, 0.0)
    cdd = max(tavg - DD_BASE, 0.0)
    return round(hdd, 4), round(cdd, 4)


async def upsert_weather(conn: asyncpg.Connection, df) -> int:  # type: ignore[no-untyped-def]
    """Upsert Meteostat Daily DataFrame rows into raw.weather_daily."""
    upserted = 0
    import pandas as pd  # noqa: PLC0415

    def _float(val: object) -> float | None:
        """Convert a value to float, returning None for any NA-like sentinel."""
        if val is None:
            return None
        try:
            if pd.isna(val):  # handles float NaN, pd.NA, pd.NaT
                return None
        except (TypeError, ValueError):
            pass
        try:
            return float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    async with conn.transaction():
        for idx, row in df.iterrows():
            obs_date: date = idx.date() if hasattr(idx, "date") else idx
            tavg = _float(row.get("tavg"))
            tmin = _float(row.get("tmin"))
            tmax = _float(row.get("tmax"))
            prcp = _float(row.get("prcp"))
            snow = _float(row.get("snow"))
            hdd, cdd = _compute_dd(tavg)

            await conn.execute(
                """
                INSERT INTO raw.weather_daily (
                    date, tavg_c, tmin_c, tmax_c, prcp_mm, snow_mm, hdd, cdd
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (date) DO UPDATE SET
                    tavg_c  = EXCLUDED.tavg_c,
                    tmin_c  = EXCLUDED.tmin_c,
                    tmax_c  = EXCLUDED.tmax_c,
                    prcp_mm = EXCLUDED.prcp_mm,
                    snow_mm = EXCLUDED.snow_mm,
                    hdd     = EXCLUDED.hdd,
                    cdd     = EXCLUDED.cdd
                """,
                obs_date, tavg, tmin, tmax, prcp, snow, hdd, cdd,
            )
            upserted += 1

    return upserted


async def run(db_url: str) -> None:
    end = datetime.now()

    print(f"Fetching Meteostat daily data {FETCH_START.date()} → {end.date()} …")
    # Provider.DAILY is meteostat's aggregated daily dataset for station KNYC0.
    # Must be specified explicitly in v2; omitting providers causes fetch() to return None.
    df = Daily(STATION_ID, FETCH_START, end, providers=[Provider.DAILY]).fetch()

    if df is None or df.empty:
        print("No data returned from Meteostat. Check station availability.")
        return

    # meteostat v2 uses 'temp' (was 'tavg') and 'snwd' (was 'snow'); normalise column names.
    rename = {}
    if "temp" in df.columns and "tavg" not in df.columns:
        rename["temp"] = "tavg"
    if "snwd" in df.columns and "snow" not in df.columns:
        rename["snwd"] = "snow"
    if rename:
        df = df.rename(columns=rename)

    # Drop rows where all weather values are NaN (station gaps).
    weather_cols = ["tavg", "tmin", "tmax", "prcp", "snow"]
    df = df.dropna(subset=[c for c in weather_cols if c in df.columns], how="all")
    print(f"  Fetched {len(df)} daily records")

    conn = await asyncpg.connect(db_url)
    await conn.execute("SET statement_timeout = 0")
    n = await upsert_weather(conn, df)
    print(f"Done. upserted={n}")
    await conn.close()


async def main() -> None:
    db_url = os.environ.get("DIRECT_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DIRECT_DATABASE_URL or DATABASE_URL is not set.")
    await run(db_url)


if __name__ == "__main__":
    asyncio.run(main())
