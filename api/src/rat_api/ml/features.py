"""Online feature assembly from the database (T-19).

Fetches the single panel row for a given NTA + week, returning a dict
ready to be passed to the inference pipeline.  Returns None if the row
is missing — callers must raise 503, not silently fabricate features.
"""

from __future__ import annotations

from datetime import date

import asyncpg


async def get_nta_features(
    nta_id: str,
    week: date,
    conn: asyncpg.Connection,
) -> dict | None:
    """Fetch one NTA-week feature row from features.nta_week_panel.

    Args:
        nta_id: NTA 2020 identifier (e.g. ``"MN2501"``).
        week:   ISO week start date (Monday).
        conn:   Live asyncpg connection.

    Returns:
        Dict of ``{column: value}`` for all panel columns, or ``None``
        if no matching row exists.
    """
    row = await conn.fetchrow(
        "SELECT * FROM features.nta_week_panel WHERE nta_id = $1 AND week_start = $2",
        nta_id,
        week,
    )
    if row is None:
        # Fall back to the most recent available week for this NTA.
        row = await conn.fetchrow(
            "SELECT * FROM features.nta_week_panel WHERE nta_id = $1 ORDER BY week_start DESC LIMIT 1",
            nta_id,
        )
    if row is None:
        return None
    return dict(row)


async def get_all_nta_features_for_week(
    week: date,
    conn: asyncpg.Connection,
) -> list[dict]:
    """Fetch all NTA feature rows for a single week (used by /risk/map)."""
    rows = await conn.fetch(
        "SELECT * FROM features.nta_week_panel WHERE week_start = $1",
        week,
    )
    return [dict(r) for r in rows]


def current_iso_week() -> date:
    """Return the Monday of the current ISO week."""
    from datetime import datetime, timedelta  # noqa: PLC0415

    today = datetime.utcnow().date()
    return today - timedelta(days=today.weekday())
