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

from meteostat import Daily, Point  # type: ignore[import-untyped]

STATION_LAT = 40.7789
STATION_LON = -73.9692
STATION_ALT = 39  # metres
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
    async with conn.transaction():
        for idx, row in df.iterrows():
            obs_date: date = idx.date() if hasattr(idx, "date") else idx
            tavg = float(row["tavg"]) if row.get("tavg") is not None and str(row.get("tavg")) != "nan" else None
            tmin = float(row["tmin"]) if row.get("tmin") is not None and str(row.get("tmin")) != "nan" else None
            tmax = float(row["tmax"]) if row.get("tmax") is not None and str(row.get("tmax")) != "nan" else None
            prcp = float(row["prcp"]) if row.get("prcp") is not None and str(row.get("prcp")) != "nan" else None
            snow = float(row["snow"]) if row.get("snow") is not None and str(row.get("snow")) != "nan" else None
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
    location = Point(STATION_LAT, STATION_LON, STATION_ALT)
    end = datetime.now()

    print(f"Fetching Meteostat daily data {FETCH_START.date()} → {end.date()} …")
    df = Daily(location, FETCH_START, end).fetch()

    if df.empty:
        print("No data returned from Meteostat. Check station availability.")
        return

    # Drop rows where all weather values are NaN (station gaps).
    weather_cols = ["tavg", "tmin", "tmax", "prcp", "snow"]
    df = df.dropna(subset=[c for c in weather_cols if c in df.columns], how="all")
    print(f"  Fetched {len(df)} daily records")

    conn = await asyncpg.connect(db_url)
    n = await upsert_weather(conn, df)
    print(f"Done. upserted={n}")
    await conn.close()


async def main() -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DATABASE_URL is not set.")
    await run(db_url)


if __name__ == "__main__":
    asyncio.run(main())
