"""Download Sentinel-2 L2A quarterly mosaics for NYC NTAs from MPC STAC (T-30).

For each NTA and each calendar quarter since 2023-Q1, downloads a cloud-masked
median composite from Sentinel-2 L2A scenes covering NYC tiles 18TWL and 18TWK.

Output layout (gitignored):
    data/sentinel2/<nta_id>/<YYYY>-Q<N>/composite.tif

The composite contains 7 bands:
    0: B02  Blue       10m
    1: B03  Green      10m
    2: B04  Red        10m
    3: B08  NIR        10m
    4: B8A  Narrow NIR 20m (resampled to 10m)
    5: B11  SWIR-1     20m (resampled to 10m)
    6: B12  SWIR-2     20m (resampled to 10m)

These are the bands expected by Clay v1.5.

Idempotency: quarters that already have a composite.tif are skipped.

Usage (from repo root)::

    uv run --package rat-ml --extra vision python ml/scripts/ingest_sentinel2.py

Optional env vars:
    PLANETARY_COMPUTER_KEY  — anonymous access works for Sentinel-2 L2A, but
                              adding a key raises rate limits.
    DIRECT_DATABASE_URL     — Supabase direct connection; used to fetch NTA
                              boundary geometries.

Requires:
    uv run --package rat-ml --extra vision  (installs rasterio, stackstac, etc.)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import date
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sentinel-2 MGRS tiles that together cover all 5 NYC boroughs.
NYC_TILES: list[str] = ["18TWL", "18TWK", "18TXL"]

# Spectral bands to download; order defines channel index in the output tif.
BANDS: list[str] = ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]

# Cloud/shadow mask: keep pixels where SCL is 4 (vegetation), 5 (bare soil),
# 6 (water), or 11 (snow) — effectively reject cloud (8,9,10) and shadow (3).
VALID_SCL: list[int] = [4, 5, 6, 11]

# Maximum cloud cover (%) for a scene to be included in the mosaic.
MAX_CLOUD_PCT: float = 20.0

# Output CRS — WGS 84 geographic, matching the rest of the pipeline.
OUTPUT_CRS: str = "EPSG:4326"

# Native Sentinel-2 resolution to resample everything to (metres, UTM).
TARGET_RESOLUTION: int = 10

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = REPO_ROOT / "data" / "sentinel2"

MPC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"


# ---------------------------------------------------------------------------
# Quarter helpers
# ---------------------------------------------------------------------------

def _quarters_since(start_year: int = 2023, start_q: int = 1) -> list[tuple[str, date, date]]:
    """Return list of (label, start_date, end_date) for quarters from start to now."""
    today = date.today()
    quarters = []
    year, q = start_year, start_q
    while True:
        q_start_month = (q - 1) * 3 + 1
        q_end_month = q_start_month + 2
        q_start = date(year, q_start_month, 1)
        # Last day of the quarter's last month
        if q_end_month == 12:
            q_end = date(year, 12, 31)
        else:
            q_end = date(year, q_end_month + 1, 1).replace(day=1)
            from datetime import timedelta
            q_end = q_end - timedelta(days=1)
        if q_start > today:
            break
        label = f"{year}-Q{q}"
        quarters.append((label, q_start, min(q_end, today)))
        q += 1
        if q > 4:
            q = 1
            year += 1
    return quarters


# ---------------------------------------------------------------------------
# NTA boundary loading
# ---------------------------------------------------------------------------

async def _load_nta_boundaries(db_url: str) -> "gpd.GeoDataFrame":
    """Fetch NTA boundaries from Supabase as a GeoDataFrame."""
    import asyncpg  # noqa: PLC0415
    import geopandas as gpd  # noqa: PLC0415
    import json  # noqa: PLC0415
    from shapely.geometry import shape  # noqa: PLC0415

    conn = await asyncpg.connect(db_url)
    try:
        rows = await conn.fetch(
            "SELECT nta_id, ST_AsGeoJSON(geom)::text AS geom_json FROM raw.nta_boundaries ORDER BY nta_id"
        )
    finally:
        await conn.close()

    return gpd.GeoDataFrame(
        {"nta_id": [r["nta_id"] for r in rows]},
        geometry=[shape(json.loads(r["geom_json"])) for r in rows],
        crs="EPSG:4326",
    )


# ---------------------------------------------------------------------------
# STAC search
# ---------------------------------------------------------------------------

def _search_scenes(
    bbox: tuple[float, float, float, float],
    date_start: date,
    date_end: date,
) -> list:
    """Search MPC STAC for Sentinel-2 L2A scenes, returning signed items."""
    import planetary_computer  # noqa: PLC0415
    import pystac_client  # noqa: PLC0415

    catalog = pystac_client.Client.open(
        MPC_STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=[COLLECTION],
        bbox=list(bbox),
        datetime=f"{date_start.isoformat()}/{date_end.isoformat()}",
        query={"eo:cloud_cover": {"lt": MAX_CLOUD_PCT}},
    )
    items = list(search.items())
    log.info("  Found %d scenes (cloud < %.0f%%)", len(items), MAX_CLOUD_PCT)
    return items


# ---------------------------------------------------------------------------
# Mosaic building
# ---------------------------------------------------------------------------

def _build_mosaic(
    items: list,
    bbox: tuple[float, float, float, float],
) -> "xr.DataArray | None":
    """Stack items into a cloud-masked median composite over the bbox.

    Returns None if no valid pixels are found.
    """
    import numpy as np  # noqa: PLC0415
    import stackstac  # noqa: PLC0415
    import xarray as xr  # noqa: PLC0415

    if not items:
        return None

    # Stack all scenes for the spectral bands + SCL mask band
    stack = stackstac.stack(
        items,
        assets=BANDS + ["SCL"],
        resolution=TARGET_RESOLUTION,
        bounds_latlon=bbox,
        dtype="float32",
        fill_value=float("nan"),
    )

    # Separate SCL from spectral bands
    scl = stack.sel(band="SCL").values  # (time, y, x)
    spectral = stack.sel(band=BANDS)    # (time, band, y, x)

    # Build valid-pixel mask from SCL (avoid Python int + array with NumPy 2.0)
    valid_mask = np.zeros(scl.shape, dtype=bool)
    for v in VALID_SCL:
        valid_mask |= (scl == v)

    # Apply mask: invalid pixels → NaN
    spectral_vals = spectral.values.copy()  # (time, band, y, x)
    spectral_vals[:, :, ~valid_mask] = float("nan")

    # Median composite along time dimension (ignores NaN)
    with np.errstate(all="ignore"):
        composite = np.nanmedian(spectral_vals, axis=0)  # (band, y, x)

    if np.all(np.isnan(composite)):
        log.warning("  All pixels are NaN after cloud masking — no valid data")
        return None

    # Rebuild as DataArray preserving spatial metadata
    result = xr.DataArray(
        composite,
        dims=["band", "y", "x"],
        coords={
            "band": BANDS,
            "y": spectral.coords["y"].values,
            "x": spectral.coords["x"].values,
        },
    )
    result = result.rio.set_crs(spectral.rio.crs)
    return result


# ---------------------------------------------------------------------------
# Per-NTA processing
# ---------------------------------------------------------------------------

def _save_composite(
    composite: "xr.DataArray",
    out_path: Path,
) -> None:
    """Reproject to WGS84 and save as GeoTIFF."""
    import numpy as np  # noqa: PLC0415

    reproj = composite.rio.reproject(OUTPUT_CRS)
    # Replace NaN with nodata value
    reproj = reproj.fillna(-9999)
    reproj.rio.to_raster(
        str(out_path),
        driver="GTiff",
        dtype="float32",
        nodata=-9999,
        compress="lzw",
    )
    log.info("  Saved composite → %s", out_path)


def process_nta(
    nta_id: str,
    bbox: tuple[float, float, float, float],
    quarter_label: str,
    date_start: date,
    date_end: date,
    out_dir: Path,
) -> bool:
    """Download and save a quarterly composite for one NTA.

    Returns True if composite was written, False if skipped or no data.
    """
    out_path = out_dir / "composite.tif"
    if out_path.exists():
        log.debug("  Skipping %s %s (already exists)", nta_id, quarter_label)
        return False

    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Processing NTA=%s quarter=%s", nta_id, quarter_label)

    items = _search_scenes(bbox, date_start, date_end)
    if not items:
        log.warning("  No scenes found for %s %s", nta_id, quarter_label)
        return False

    composite = _build_mosaic(items, bbox)
    if composite is None:
        return False

    _save_composite(composite, out_path)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(db_url: str, start_year: int = 2023, start_q: int = 1) -> dict[str, int]:
    """Download quarterly mosaics for all NTAs.

    Returns:
        dict with keys 'written', 'skipped', 'failed'.
    """
    gdf = await _load_nta_boundaries(db_url)
    log.info("Loaded %d NTA boundaries", len(gdf))

    quarters = _quarters_since(start_year, start_q)
    log.info("Processing %d quarters from %s-Q%d", len(quarters), start_year, start_q)

    written = skipped = failed = 0

    for nta_id, row in gdf.set_index("nta_id").iterrows():
        geom = row.geometry
        minx, miny, maxx, maxy = geom.bounds
        # Small buffer (0.005 deg ≈ 500m) to ensure full coverage
        bbox = (minx - 0.005, miny - 0.005, maxx + 0.005, maxy + 0.005)

        for quarter_label, date_start, date_end in quarters:
            nta_dir = DATA_DIR / str(nta_id) / quarter_label
            try:
                result = process_nta(
                    str(nta_id), bbox, quarter_label, date_start, date_end, nta_dir
                )
                if result:
                    written += 1
                else:
                    skipped += 1
            except Exception as exc:  # noqa: BLE001
                log.error("Failed NTA=%s quarter=%s: %s", nta_id, quarter_label, exc)
                failed += 1

    return {"written": written, "skipped": skipped, "failed": failed}


async def main() -> None:
    db_url = os.environ.get("DIRECT_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not db_url:
        sys.exit("DIRECT_DATABASE_URL is not set. Add it to .env or export it.")

    pc_key = os.environ.get("PLANETARY_COMPUTER_KEY", "")
    if pc_key:
        import planetary_computer  # noqa: PLC0415
        planetary_computer.set_subscription_key(pc_key)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    log.info("Starting Sentinel-2 ingest …")
    stats = await run(db_url)
    log.info(
        "Done. written=%d skipped=%d failed=%d",
        stats["written"], stats["skipped"], stats["failed"],
    )


if __name__ == "__main__":
    asyncio.run(main())
