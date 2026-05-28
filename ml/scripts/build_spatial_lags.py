"""Populate spatial-lag columns in features.nta_week_panel (T-09).

Builds a queen-contiguity adjacency matrix from raw.nta_boundaries and
updates neighbor_active_rat_signs_rate_lag_1w and
neighbor_complaints_count_lag_4w for all rows in the panel.

Usage (from repo root)::

    uv run --package rat-ml python ml/scripts/build_spatial_lags.py

Requires environment variables (or .env file):
    DIRECT_DATABASE_URL — direct Supabase connection (bypasses PgBouncer)
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

from rat_ml.features.spatial_lags import run


async def main() -> None:
    db_url = os.environ.get("DIRECT_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DIRECT_DATABASE_URL is not set. Add it to .env or export it.")

    print("Building spatial-lag features …")
    n = await run(db_url)
    print(f"Done. rows updated={n}")


if __name__ == "__main__":
    asyncio.run(main())
