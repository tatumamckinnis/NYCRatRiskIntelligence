"""Assemble features.nta_week_panel from all raw sources (T-08).

Runs six assembly steps in a single multi-CTE INSERT … ON CONFLICT upsert:

  1. Labels      — rodent inspection counts per NTA-week via PostGIS spatial join
  2. 311 lags    — complaint counts + 1w/4w/12w LAG window functions
  3. Rest. pest  — restaurant pest-violation counts via BBL→PLUTO→NTA
  4. DOB permits — permit and demolition counts via BBL→PLUTO→NTA
  5. Weather     — weekly averages/totals from raw.weather_daily
  6. PLUTO static — units_total, year_built_median, landuse pcts from raw.pluto

SQL logic lives in rat_ml.features.panel; this script is the CLI entry point.

Usage (from repo root)::

    uv run --package rat-ml python ml/scripts/build_panel.py
"""

from __future__ import annotations

import asyncio
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

from rat_ml.features.panel import run
from rat_ml.reporting.data_quality import run as run_dq_report


async def main() -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DATABASE_URL is not set.")

    print("Building features.nta_week_panel …")
    n = await run(db_url)
    print(f"Done. rows upserted={n}")

    print("Generating data quality report …")
    path = await run_dq_report(db_url)
    print(f"Report written to {path}")


if __name__ == "__main__":
    asyncio.run(main())
