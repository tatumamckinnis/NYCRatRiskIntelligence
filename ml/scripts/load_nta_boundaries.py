#!/usr/bin/env python3
"""Load NTA 2020 boundary GeoJSON from NYC DCP into raw.nta_boundaries.

Also downloads the 2010→2020 NTA crosswalk CSV to data/ for use by
ml/src/rat_ml/data/tract_crosswalk.py.

Data sources (public, no API key required):
  NTA 2020 boundaries — NYC Open Data dataset 9nt8-h7nd
  NTA crosswalk CSV   — NYC DCP 2020 Census tabulation equivalency file

Usage (from repo root):
    uv run --package rat-ml python ml/scripts/load_nta_boundaries.py
    uv run --package rat-ml python ml/scripts/load_nta_boundaries.py \\
        --nta-url <url> --crosswalk-url <url>

Requires environment variables (or .env file):
    DIRECT_DATABASE_URL — direct Supabase connection (bypasses PgBouncer)
"""

from __future__ import annotations

import argparse
import asyncio
import io
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

import asyncpg
import geopandas as gpd
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Default data source URLs.
# NTA 2020 GeoJSON:  NYC Open Data dataset 9nt8-h7nd (NYC DCP, 2020 NTAs).
#   Verify at: https://data.cityofnewyork.us/City-Government/
#              2020-Neighborhood-Tabulation-Areas-NTAs-/9nt8-h7nd
# NTA crosswalk CSV: NYC DCP 2020 Census tabulation equivalency file.
#   Verify at: https://www.nyc.gov/site/planning/data-maps/
#              open-data/census-download-requests.page
# ---------------------------------------------------------------------------
DEFAULT_NTA_URL = (
    "https://data.cityofnewyork.us/api/geospatial/9nt8-h7nd"
    "?method=export&type=GeoJSON"
)
DEFAULT_CROSSWALK_URL = (
    "https://www.nyc.gov/assets/planning/download/office/data-maps/"
    "nyc-population/census2020/nyc2020census_tabulation_equiv.xlsx"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

# Expected CRS of the source file; re-project to WGS84 if different.
WGS84 = "EPSG:4326"
NY_STATE_PLANE = "EPSG:2263"


def download_nta_boundaries(url: str) -> gpd.GeoDataFrame:
    print(f"Downloading NTA 2020 boundaries from {url} …")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    gdf = gpd.read_file(io.BytesIO(resp.content))
    print(f"  Read {len(gdf)} NTA features, CRS={gdf.crs}")

    # Re-project to WGS84 if necessary (source is sometimes NY State Plane).
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        print(f"  Re-projecting {gdf.crs} → {WGS84}")
        gdf = gdf.to_crs(WGS84)

    return gdf


def download_crosswalk(url: str, dest: Path) -> Path:
    print(f"Downloading NTA crosswalk from {url} …")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    suffix = Path(url.split("?")[0]).suffix or ".xlsx"
    dest_file = dest / f"nta_2010_2020_crosswalk{suffix}"
    dest_file.write_bytes(resp.content)
    print(f"  Saved to {dest_file}")
    return dest_file


def normalize_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Normalize column names to the raw.nta_boundaries schema."""
    # NYC Open Data NTA 2020 column names (lowercase after normalization).
    col_map: dict[str, str] = {}
    lower_cols = {c.lower(): c for c in gdf.columns}

    for candidate, target in [
        (["ntacode", "nta_code", "nta2020", "ntacode20"], "nta_id"),
        (["ntaname", "nta_name", "name"], "nta_name"),
        (["boroname", "borough", "boro_name", "borocode"], "borough"),
        (["shape_area", "area", "shape__area"], "area_sq_m"),
    ]:
        for c in candidate:
            if c in lower_cols:
                col_map[lower_cols[c]] = target
                break

    gdf = gdf.rename(columns=col_map)

    # Compute area in square metres from the geometry if not present.
    if "area_sq_m" not in gdf.columns:
        projected = gdf.to_crs("EPSG:6539")  # NYC Long Island projection, metres
        gdf["area_sq_m"] = projected.geometry.area

    return gdf[["nta_id", "nta_name", "borough", "geometry", "area_sq_m"]]


async def upsert_boundaries(gdf: gpd.GeoDataFrame, db_url: str) -> int:
    """Upsert NTA boundaries into raw.nta_boundaries. Returns row count upserted."""
    conn = await asyncpg.connect(db_url)
    try:
        rows_upserted = 0
        async with conn.transaction():
            for _, row in gdf.iterrows():
                geom_wkt = row.geometry.wkt if row.geometry else None
                await conn.execute(
                    """
                    INSERT INTO raw.nta_boundaries (nta_id, nta_name, borough, geom, area_sq_m)
                    VALUES ($1, $2, $3, ST_SetSRID(ST_GeomFromText($4), 4326), $5)
                    ON CONFLICT (nta_id) DO UPDATE SET
                        nta_name  = EXCLUDED.nta_name,
                        borough   = EXCLUDED.borough,
                        geom      = EXCLUDED.geom,
                        area_sq_m = EXCLUDED.area_sq_m
                    """,
                    str(row["nta_id"]),
                    str(row["nta_name"]),
                    str(row["borough"]),
                    geom_wkt,
                    float(row["area_sq_m"]) if row["area_sq_m"] is not None else None,
                )
                rows_upserted += 1
        return rows_upserted
    finally:
        await conn.close()


async def main(nta_url: str, crosswalk_url: str) -> None:
    db_url = os.environ.get("DIRECT_DATABASE_URL")
    if not db_url:
        sys.exit("DIRECT_DATABASE_URL is not set. Add it to .env or export it.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download and load NTA 2020 boundaries.
    gdf = download_nta_boundaries(nta_url)
    gdf = normalize_gdf(gdf)
    print(f"  Normalized to {len(gdf)} rows with columns {list(gdf.columns)}")

    # 2. Save a local copy (gitignored).
    local_geojson = DATA_DIR / "nta_2020_boundaries.geojson"
    gdf.to_file(local_geojson, driver="GeoJSON")
    print(f"  Cached to {local_geojson}")

    # 3. Upsert into Supabase.
    n = await upsert_boundaries(gdf, db_url)
    print(f"  Upserted {n} rows into raw.nta_boundaries")

    # 4. Download crosswalk for tract_crosswalk.py.
    download_crosswalk(crosswalk_url, DATA_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nta-url", default=DEFAULT_NTA_URL,
                        help="GeoJSON download URL for NTA 2020 boundaries")
    parser.add_argument("--crosswalk-url", default=DEFAULT_CROSSWALK_URL,
                        help="Download URL for 2010→2020 NTA crosswalk file")
    args = parser.parse_args()
    asyncio.run(main(args.nta_url, args.crosswalk_url))
