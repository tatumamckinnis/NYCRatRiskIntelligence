"""Tests for spatial-lag feature engineering (T-09).

Unit tests: adjacency logic and SQL generation, no DB required.
Integration tests (pytest.mark.integration): live adjacency symmetry check
and Manhattan cross-borough isolation test.
"""

from __future__ import annotations

import os

import geopandas as gpd
import pytest
from shapely.geometry import box

from rat_ml.features.spatial_lags import build_queen_adjacency, build_spatial_lags_sql


# ===========================================================================
# Helpers
# ===========================================================================

def _row_gdf() -> gpd.GeoDataFrame:
    """Three touching unit squares in a row: A — B — C."""
    return gpd.GeoDataFrame(
        {"nta_id": ["A", "B", "C"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        crs="EPSG:4326",
    )


def _isolated_gdf() -> gpd.GeoDataFrame:
    """Two squares far apart — neither is a queen neighbor of the other."""
    return gpd.GeoDataFrame(
        {"nta_id": ["X", "Y"]},
        geometry=[box(0, 0, 1, 1), box(10, 10, 11, 11)],
        crs="EPSG:4326",
    )


# ===========================================================================
# Unit tests — adjacency logic
# ===========================================================================

def test_adjacency_symmetric() -> None:
    adj = build_queen_adjacency(_row_gdf())
    for nta_id, neighbors in adj.items():
        for nbr in neighbors:
            assert nta_id in adj[nbr], (
                f"Adjacency not symmetric: {nta_id!r} lists {nbr!r} as neighbor "
                f"but {nbr!r} does not list {nta_id!r}"
            )


def test_adjacency_middle_has_two_neighbors() -> None:
    adj = build_queen_adjacency(_row_gdf())
    assert set(adj["B"]) == {"A", "C"}


def test_adjacency_endpoints_have_one_neighbor() -> None:
    adj = build_queen_adjacency(_row_gdf())
    assert adj["A"] == ["B"]
    assert adj["C"] == ["B"]


def test_adjacency_nonadjacent_excluded() -> None:
    """A and C share no edge or corner — must not appear as neighbors."""
    adj = build_queen_adjacency(_row_gdf())
    assert "C" not in adj["A"]
    assert "A" not in adj["C"]


def test_adjacency_isolated_has_no_neighbors() -> None:
    adj = build_queen_adjacency(_isolated_gdf())
    assert adj["X"] == []
    assert adj["Y"] == []


# ===========================================================================
# Unit tests — SQL generation
# ===========================================================================

def test_spatial_lags_sql_updates_both_columns() -> None:
    sql = build_spatial_lags_sql()
    assert "neighbor_active_rat_signs_rate_lag_1w" in sql
    assert "neighbor_complaints_count_lag_4w" in sql


def test_spatial_lags_sql_uses_avg() -> None:
    assert "AVG(" in build_spatial_lags_sql()


def test_spatial_lags_sql_references_temp_table() -> None:
    assert "tmp_nta_adjacency" in build_spatial_lags_sql()


def test_spatial_lags_sql_uses_interval_7_days() -> None:
    """Rate lag must look back exactly one week."""
    assert "INTERVAL '7 days'" in build_spatial_lags_sql()


def test_spatial_lags_sql_is_update() -> None:
    assert build_spatial_lags_sql().strip().upper().startswith("UPDATE")


# ===========================================================================
# Integration tests
# ===========================================================================

@pytest.fixture()
async def db_conn():
    import asyncpg  # noqa: PLC0415
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    conn = await asyncpg.connect(url)
    try:
        yield conn
    finally:
        await conn.close()


@pytest.mark.integration
async def test_adjacency_symmetric_live(db_conn) -> None:
    """Queen adjacency built from real NTA boundaries must be symmetric."""
    from rat_ml.features.spatial_lags import _fetch_boundaries  # noqa: PLC0415

    gdf = await _fetch_boundaries(db_conn)
    adj = build_queen_adjacency(gdf)

    violations = [
        (a, b)
        for a, neighbors in adj.items()
        for b in neighbors
        if a not in adj.get(b, [])
    ]
    assert not violations, f"Asymmetric adjacency pairs (first 5): {violations[:5]}"


@pytest.mark.integration
async def test_manhattan_no_cross_borough_neighbors(db_conn) -> None:
    """Manhattan NTAs (MN prefix) must not border Bronx/Brooklyn/Queens/SI.

    Manhattan is an island; queen contiguity on land polygons should produce
    no cross-water neighbors.
    """
    from rat_ml.features.spatial_lags import _fetch_boundaries  # noqa: PLC0415

    gdf = await _fetch_boundaries(db_conn)
    adj = build_queen_adjacency(gdf)

    other_prefixes = ("BX", "BK", "QN", "SI")
    violations = [
        (mn, nbr)
        for mn, neighbors in adj.items()
        if mn.startswith("MN")
        for nbr in neighbors
        if any(nbr.startswith(p) for p in other_prefixes)
    ]
    assert not violations, (
        f"Manhattan NTAs have cross-borough neighbors: {violations[:5]}"
    )


@pytest.mark.integration
async def test_spatial_lags_run_updates_rows() -> None:
    """run() must report updating at least one row."""
    from rat_ml.features.spatial_lags import run  # noqa: PLC0415

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        pytest.skip("DATABASE_URL not set")

    n = await run(db_url)
    assert n > 0, "Expected run() to update at least one row in nta_week_panel"
