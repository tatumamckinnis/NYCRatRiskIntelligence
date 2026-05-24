"""Regime-indicator feature engineering.

Populates five boolean columns in features.nta_week_panel that flag
NYC policy regimes affecting rat activity and reporting patterns:

- regime_covid:                     2020-03-01 – 2020-06-30
- regime_8pm_setout:                2023-04-01 onward
- regime_commercial_containerization: 2024-03-01 onward (full implementation)
- regime_residential_containerization: 2024-11-01 onward
- regime_rmz_active:                FALSE for all rows in Phase 1
                                    (RMZ data not yet ingested)

All thresholds are applied to week_start (Monday of the ISO week).

Usage (from repo root)::

    uv run --package rat-ml python ml/scripts/build_regime_indicators.py
"""

from __future__ import annotations

import asyncpg


_UPDATE_SQL = """
UPDATE features.nta_week_panel
SET
    regime_covid                        = week_start BETWEEN '2020-03-01' AND '2020-06-30',
    regime_8pm_setout                   = week_start >= '2023-04-01',
    regime_commercial_containerization  = week_start >= '2024-03-01',
    regime_residential_containerization = week_start >= '2024-11-01',
    regime_rmz_active                   = FALSE
"""


def build_regime_sql() -> str:
    """Return the UPDATE SQL that populates all five regime columns."""
    return _UPDATE_SQL


async def run(db_url: str) -> int:
    """Populate regime indicator columns and return the number of rows updated."""
    conn = await asyncpg.connect(db_url)
    try:
        result = await conn.execute(build_regime_sql())
        parts = result.split()
        return int(parts[-1]) if parts else 0
    finally:
        await conn.close()
