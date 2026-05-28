"""Panel assembly: build features.nta_week_panel from raw sources.

Used by ml/scripts/build_panel.py; separated into this module so the
SQL-building logic is importable from tests without invoking the CLI.

Assembly steps
--------------
1. Labels      — rodent inspection counts per NTA-week via PostGIS ST_Within
2. 311 lags    — complaint counts + 1w/4w/12w LAG window functions
3. Rest. pest  — restaurant pest-violation counts via BBL→PLUTO→NTA
4. DOB permits — permit and demolition counts via BBL→PLUTO→NTA
5. Weather     — weekly averages/totals from raw.weather_daily
6. PLUTO static — units_total, year_built_median, landuse pcts from raw.pluto
"""

from __future__ import annotations

import asyncpg

from rat_ml.features.temporal import COMMERCIAL_LANDUSE, RESIDENTIAL_LANDUSE


def _landuse_pct_expr(codes: tuple[str, ...], alias: str) -> str:
    """SQL expression for fraction of lots whose landuse is in *codes*."""
    in_list = ", ".join(f"'{c}'" for c in codes)
    return (
        f"ROUND(\n"
        f"            SUM(CASE WHEN landuse IN ({in_list}) THEN 1 ELSE 0 END)::NUMERIC\n"
        f"            / NULLIF(COUNT(*), 0), 4\n"
        f"        ) AS {alias}"
    )


def build_panel_sql() -> str:
    """Return the full INSERT … ON CONFLICT upsert SQL for the panel."""
    res_pct = _landuse_pct_expr(RESIDENTIAL_LANDUSE, "landuse_residential_pct")
    com_pct = _landuse_pct_expr(COMMERCIAL_LANDUSE, "landuse_commercial_pct")

    return f"""
WITH
-- -----------------------------------------------------------------------
-- Step 1 — Labels
-- PostGIS spatial join: assign each inspection to the NTA whose boundary
-- contains its point, then aggregate to NTA-week counts.
-- -----------------------------------------------------------------------
labels AS (
    SELECT
        b.nta_id,
        date_trunc('week', i.inspection_date)::date AS week_start,
        COUNT(*)                                           AS inspections_count,
        COUNT(*) FILTER (WHERE i.result IN (
            'Failed for Rat Activity',
            'Failed for Rat Activity and Other Reason'
        ))                                                 AS active_rat_signs_count
    FROM raw.rodent_inspections i
    JOIN raw.nta_boundaries b ON ST_Within(i.geom, b.geom)
    WHERE i.geom IS NOT NULL
      AND i.inspection_date <= CURRENT_DATE
    GROUP BY b.nta_id, date_trunc('week', i.inspection_date)::date
),

-- -----------------------------------------------------------------------
-- Step 2 — 311 complaint counts and lags
-- Window functions over raw.complaints_nta_week; each NTA has its own
-- independent lag series (PARTITION BY nta_id).
-- -----------------------------------------------------------------------
complaint_lags AS (
    SELECT
        nta_id,
        week_start,
        complaint_count,
        LAG(complaint_count,  1) OVER (PARTITION BY nta_id ORDER BY week_start) AS lag_1w,
        LAG(complaint_count,  4) OVER (PARTITION BY nta_id ORDER BY week_start) AS lag_4w,
        LAG(complaint_count, 12) OVER (PARTITION BY nta_id ORDER BY week_start) AS lag_12w
    FROM raw.complaints_nta_week
),

-- -----------------------------------------------------------------------
-- Step 3 — Restaurant pest violations
-- BBL→PLUTO→nta2020 (Option B): join on normalized BBL, group to NTA-week.
-- Only counts rows where is_pest_violation = TRUE.
-- -----------------------------------------------------------------------
rest_pest AS (
    SELECT
        p.nta2020                                           AS nta_id,
        date_trunc('week', r.inspection_date)::date         AS week_start,
        COUNT(*) FILTER (WHERE r.is_pest_violation)         AS rest_pest_violations_count
    FROM raw.restaurant_inspections r
    JOIN raw.pluto p ON r.bbl = p.bbl
    WHERE p.nta2020 IS NOT NULL
    GROUP BY p.nta2020, date_trunc('week', r.inspection_date)::date
),

-- -----------------------------------------------------------------------
-- Step 4 — DOB permits
-- BBL→PLUTO→nta2020 (Option B): permits issued in each NTA-week.
-- demolitions_count: job_type = 'DM' permits only.
-- -----------------------------------------------------------------------
dob AS (
    SELECT
        p.nta2020                                               AS nta_id,
        date_trunc('week', d.issuance_date)::date               AS week_start,
        COUNT(*)                                                AS permits_active_count,
        COUNT(*) FILTER (WHERE d.job_type = 'DM')              AS demolitions_count
    FROM raw.dob_permits d
    JOIN raw.pluto p ON d.bbl = p.bbl
    WHERE p.nta2020 IS NOT NULL
      AND d.issuance_date IS NOT NULL
    GROUP BY p.nta2020, date_trunc('week', d.issuance_date)::date
),

-- -----------------------------------------------------------------------
-- Step 5 — Weather weekly aggregates
-- Align to Monday week_start (same convention as rest of panel).
-- tavg_c averaged; prcp_mm / hdd / cdd summed across the 7 days.
-- -----------------------------------------------------------------------
weather AS (
    SELECT
        date_trunc('week', date)::date  AS week_start,
        ROUND(AVG(tavg_c)::NUMERIC, 4)  AS weather_tavg_c,
        ROUND(SUM(prcp_mm)::NUMERIC, 4) AS weather_prcp_mm,
        ROUND(SUM(hdd)::NUMERIC, 4)     AS weather_hdd,
        ROUND(SUM(cdd)::NUMERIC, 4)     AS weather_cdd
    FROM raw.weather_daily
    GROUP BY date_trunc('week', date)::date
),

-- -----------------------------------------------------------------------
-- Step 6 — PLUTO static aggregates per NTA
-- One row per NTA; LEFT JOINed (not grouped) by week so values repeat
-- across all weeks for the same NTA.
-- -----------------------------------------------------------------------
pluto_static AS (
    SELECT
        nta2020                                              AS nta_id,
        SUM(unitstotal)                                      AS units_total,
        PERCENTILE_CONT(0.5) WITHIN GROUP (
            ORDER BY yearbuilt
        )::INTEGER                                           AS year_built_median,
        {res_pct},
        {com_pct}
    FROM raw.pluto
    WHERE nta2020 IS NOT NULL
    GROUP BY nta2020
)

-- -----------------------------------------------------------------------
-- Final upsert — labels is the spine; everything else left-joins
-- -----------------------------------------------------------------------
INSERT INTO features.nta_week_panel (
    nta_id,
    week_start,
    active_rat_signs_count,
    inspections_count,
    active_rat_signs_rate,
    active_rat_signs_ind,
    complaints_count,
    complaints_lag_1w,
    complaints_lag_4w,
    complaints_lag_12w,
    rest_pest_violations_count,
    permits_active_count,
    demolitions_count,
    weather_tavg_c,
    weather_prcp_mm,
    weather_hdd,
    weather_cdd,
    units_total,
    year_built_median,
    landuse_residential_pct,
    landuse_commercial_pct
)
SELECT
    l.nta_id,
    l.week_start,
    l.active_rat_signs_count,
    l.inspections_count,
    ROUND(
        l.active_rat_signs_count::NUMERIC / NULLIF(l.inspections_count, 0),
        4
    )                                                AS active_rat_signs_rate,
    l.active_rat_signs_count > 0                     AS active_rat_signs_ind,
    COALESCE(cl.complaint_count, 0)                  AS complaints_count,
    cl.lag_1w                                        AS complaints_lag_1w,
    cl.lag_4w                                        AS complaints_lag_4w,
    cl.lag_12w                                       AS complaints_lag_12w,
    COALESCE(rp.rest_pest_violations_count, 0)       AS rest_pest_violations_count,
    COALESCE(d.permits_active_count, 0)              AS permits_active_count,
    COALESCE(d.demolitions_count, 0)                 AS demolitions_count,
    w.weather_tavg_c,
    w.weather_prcp_mm,
    w.weather_hdd,
    w.weather_cdd,
    ps.units_total,
    ps.year_built_median,
    ps.landuse_residential_pct,
    ps.landuse_commercial_pct
FROM labels l
LEFT JOIN complaint_lags cl ON cl.nta_id = l.nta_id AND cl.week_start = l.week_start
LEFT JOIN rest_pest      rp ON rp.nta_id = l.nta_id AND rp.week_start = l.week_start
LEFT JOIN dob             d ON  d.nta_id = l.nta_id AND  d.week_start = l.week_start
LEFT JOIN weather         w ON  w.week_start = l.week_start
LEFT JOIN pluto_static   ps ON ps.nta_id = l.nta_id
ON CONFLICT (nta_id, week_start) DO UPDATE SET
    active_rat_signs_count      = EXCLUDED.active_rat_signs_count,
    inspections_count           = EXCLUDED.inspections_count,
    active_rat_signs_rate       = EXCLUDED.active_rat_signs_rate,
    active_rat_signs_ind        = EXCLUDED.active_rat_signs_ind,
    complaints_count            = EXCLUDED.complaints_count,
    complaints_lag_1w           = EXCLUDED.complaints_lag_1w,
    complaints_lag_4w           = EXCLUDED.complaints_lag_4w,
    complaints_lag_12w          = EXCLUDED.complaints_lag_12w,
    rest_pest_violations_count  = EXCLUDED.rest_pest_violations_count,
    permits_active_count        = EXCLUDED.permits_active_count,
    demolitions_count           = EXCLUDED.demolitions_count,
    weather_tavg_c              = EXCLUDED.weather_tavg_c,
    weather_prcp_mm             = EXCLUDED.weather_prcp_mm,
    weather_hdd                 = EXCLUDED.weather_hdd,
    weather_cdd                 = EXCLUDED.weather_cdd,
    units_total                 = EXCLUDED.units_total,
    year_built_median           = EXCLUDED.year_built_median,
    landuse_residential_pct     = EXCLUDED.landuse_residential_pct,
    landuse_commercial_pct      = EXCLUDED.landuse_commercial_pct
"""


async def run(db_url: str) -> int:
    """Execute the panel assembly upsert and return the number of rows affected."""
    sql = build_panel_sql()
    conn = await asyncpg.connect(db_url)
    try:
        result = await conn.execute(sql)
        parts = result.split()
        return int(parts[-1]) if parts else 0
    finally:
        await conn.close()
