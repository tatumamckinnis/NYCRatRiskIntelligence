"""SQL window-function fragments for time-series feature engineering.

All fragments assume the target table has columns `nta_id` (TEXT) and
`week_start` (DATE) — the grain of features.nta_week_panel.  Partitioning
by nta_id ensures each NTA gets its own independent lag series.

Usage in build_panel.py::

    from rat_ml.features.temporal import complaints_lags_cte, lag_sql

    cte = complaints_lags_cte()
    sql = f"WITH complaint_lags AS ({cte}) UPDATE ..."
"""

from __future__ import annotations


def lag_sql(col: str, weeks: int, alias: str | None = None) -> str:
    """Return a LAG window-function SQL fragment.

    Partitions by nta_id, orders by week_start ascending so each NTA has
    its own lag series without bleed-across from neighbouring NTAs.

    Args:
        col:   Column name to lag (no table prefix, no quoting).
        weeks: Number of weekly periods to look back.
        alias: Output alias; defaults to ``<col>_lag_<weeks>w``.

    Returns:
        SQL fragment suitable for use in a SELECT list, e.g.::

            LAG(complaint_count, 4) OVER (PARTITION BY nta_id ORDER BY week_start) AS complaint_count_lag_4w

    Example::

        >>> lag_sql("complaint_count", 4)
        'LAG(complaint_count, 4) OVER (PARTITION BY nta_id ORDER BY week_start) AS complaint_count_lag_4w'
        >>> lag_sql("complaint_count", 1, alias="lag_1w")
        'LAG(complaint_count, 1) OVER (PARTITION BY nta_id ORDER BY week_start) AS lag_1w'
    """
    _alias = alias or f"{col}_lag_{weeks}w"
    return (
        f"LAG({col}, {weeks}) "
        f"OVER (PARTITION BY nta_id ORDER BY week_start) "
        f"AS {_alias}"
    )


def complaints_lags_cte(source: str = "raw.complaints_nta_week") -> str:
    """Return the body of a complaints-lag CTE (without the leading WITH keyword).

    Produces three lag columns — 1w, 4w, 12w — over the complaint count
    series for each NTA, ordered by week_start.

    Args:
        source: Fully-qualified Postgres table name for the complaint source.
                Defaults to ``raw.complaints_nta_week``.

    Returns:
        SQL string intended to be embedded as::

            WITH complaint_lags AS (
                <returned string>
            )

    Example output::

        SELECT
            nta_id,
            week_start,
            complaint_count,
            LAG(complaint_count, 1)  OVER (PARTITION BY nta_id ORDER BY week_start) AS lag_1w,
            LAG(complaint_count, 4)  OVER (PARTITION BY nta_id ORDER BY week_start) AS lag_4w,
            LAG(complaint_count, 12) OVER (PARTITION BY nta_id ORDER BY week_start) AS lag_12w
        FROM raw.complaints_nta_week
    """
    lag_1w  = lag_sql("complaint_count",  1, "lag_1w")
    lag_4w  = lag_sql("complaint_count",  4, "lag_4w")
    lag_12w = lag_sql("complaint_count", 12, "lag_12w")

    return f"""
        SELECT
            nta_id,
            week_start,
            complaint_count,
            {lag_1w},
            {lag_4w},
            {lag_12w}
        FROM {source}
    """


# ---------------------------------------------------------------------------
# NYC landuse code sets used in PLUTO static aggregations (step 6)
# ---------------------------------------------------------------------------

# Landuse codes considered "residential" for landuse_residential_pct.
RESIDENTIAL_LANDUSE = ("01", "02", "03")  # 1-2 family, multi-family walk-up/elevator

# Landuse codes considered "commercial" for landuse_commercial_pct.
COMMERCIAL_LANDUSE = ("04", "05", "06")   # mixed res/commercial, commercial, industrial
