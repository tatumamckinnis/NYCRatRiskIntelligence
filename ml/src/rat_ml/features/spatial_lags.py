"""Spatial-lag feature engineering: queen-contiguity neighbor averages.

Populates two columns in features.nta_week_panel that capture spatial
autocorrelation across contiguous NTAs:

- neighbor_active_rat_signs_rate_lag_1w: mean of contiguous neighbors'
  active_rat_signs_rate from the prior week.
- neighbor_complaints_count_lag_4w: mean of contiguous neighbors'
  complaints_lag_4w at the same week.

Adjacency is built once from raw.nta_boundaries using queen contiguity
(libpysal), then written to a session temp table so a single SQL UPDATE
can compute both aggregates in one pass.

Usage (from repo root)::

    uv run --package rat-ml python ml/scripts/build_spatial_lags.py
"""

from __future__ import annotations

import json

import asyncpg
import geopandas as gpd
from shapely.geometry import shape


def build_queen_adjacency(
    gdf: gpd.GeoDataFrame,
    id_col: str = "nta_id",
) -> dict[str, list[str]]:
    """Return a queen-contiguity adjacency dict from a GeoDataFrame.

    Args:
        gdf:    GeoDataFrame with a geometry column and an id column.
        id_col: Name of the column containing NTA identifiers.

    Returns:
        Dict mapping each NTA id to a (possibly empty) list of contiguous
        neighbor NTA ids.  The result is always symmetric: if A→B then B→A.

    Example::

        >>> adj = build_queen_adjacency(gdf)
        >>> "BK0101" in adj
        True
    """
    from libpysal.weights import Queen  # noqa: PLC0415

    w = Queen.from_dataframe(gdf, use_index=False, silence_warnings=True)
    ids = gdf[id_col].tolist()

    return {
        nta_id: [ids[j] for j in w.neighbors[i]]
        for i, nta_id in enumerate(ids)
    }


async def _fetch_boundaries(conn: asyncpg.Connection) -> gpd.GeoDataFrame:
    """Pull NTA boundaries from raw.nta_boundaries as a GeoDataFrame."""
    rows = await conn.fetch(
        "SELECT nta_id, ST_AsGeoJSON(geom)::text AS geom_json"
        " FROM raw.nta_boundaries"
        " ORDER BY nta_id"
    )
    return gpd.GeoDataFrame(
        {"nta_id": [r["nta_id"] for r in rows]},
        geometry=[shape(json.loads(r["geom_json"])) for r in rows],
        crs="EPSG:4326",
    )


def build_spatial_lags_sql() -> str:
    """Return the UPDATE SQL that populates the two neighbor spatial-lag columns.

    Assumes a temporary table ``tmp_nta_adjacency (nta_id TEXT, neighbor_nta_id TEXT)``
    already exists in the session (created by :func:`run`).

    - ``neighbor_active_rat_signs_rate_lag_1w``: AVG of neighbors' rate at
      ``week_start - 7 days`` (one-week temporal lag).
    - ``neighbor_complaints_count_lag_4w``: AVG of neighbors' ``complaints_lag_4w``
      column at the same ``week_start`` (spatial lag of an existing temporal lag).
    """
    return """
UPDATE features.nta_week_panel p
SET
    neighbor_active_rat_signs_rate_lag_1w = agg.neighbor_rate,
    neighbor_complaints_count_lag_4w      = agg.neighbor_complaints
FROM (
    SELECT
        cur.nta_id,
        cur.week_start,
        AVG(lag1.active_rat_signs_rate) AS neighbor_rate,
        AVG(cur_n.complaints_lag_4w)    AS neighbor_complaints
    FROM features.nta_week_panel cur
    JOIN tmp_nta_adjacency adj
        ON adj.nta_id = cur.nta_id
    LEFT JOIN features.nta_week_panel lag1
        ON lag1.nta_id     = adj.neighbor_nta_id
       AND lag1.week_start = cur.week_start - INTERVAL '7 days'
    LEFT JOIN features.nta_week_panel cur_n
        ON cur_n.nta_id     = adj.neighbor_nta_id
       AND cur_n.week_start = cur.week_start
    GROUP BY cur.nta_id, cur.week_start
) agg
WHERE p.nta_id     = agg.nta_id
  AND p.week_start = agg.week_start
"""


async def run(db_url: str) -> int:
    """Build queen adjacency from DB boundaries and update spatial-lag columns.

    Returns:
        Number of rows updated in features.nta_week_panel.
    """
    conn = await asyncpg.connect(db_url)
    try:
        gdf = await _fetch_boundaries(conn)
        adj = build_queen_adjacency(gdf)

        pairs = [
            (nta_id, neighbor)
            for nta_id, neighbors in adj.items()
            for neighbor in neighbors
        ]

        await conn.execute("DROP TABLE IF EXISTS tmp_nta_adjacency")
        await conn.execute(
            """
            CREATE TEMP TABLE tmp_nta_adjacency (
                nta_id          TEXT NOT NULL,
                neighbor_nta_id TEXT NOT NULL
            )
            """
        )
        await conn.copy_records_to_table(
            "tmp_nta_adjacency",
            records=pairs,
            columns=["nta_id", "neighbor_nta_id"],
        )

        result = await conn.execute(build_spatial_lags_sql())
        parts = result.split()
        return int(parts[-1]) if parts else 0
    finally:
        await conn.close()
