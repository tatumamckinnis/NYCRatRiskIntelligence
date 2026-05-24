"""GET /inspections/* endpoints (T-20)."""

from __future__ import annotations

from datetime import date

import asyncpg
from fastapi import APIRouter, Query, Request

from rat_api.config import get_settings
from rat_api.models.risk import InspectionItem

router = APIRouter(prefix="/inspections")


@router.get("/nta/{nta_id}", response_model=list[InspectionItem])
async def get_inspections(
    nta_id: str,
    request: Request,
    since: date = Query(
        default=None,
        description="Return inspections on or after this date. Defaults to 90 days ago.",
    ),
) -> list[InspectionItem]:
    """Return recent rodent inspection outcomes for one NTA.

    Uses a PostGIS spatial join to filter inspections whose point falls within
    the NTA boundary.
    """
    from datetime import datetime, timedelta  # noqa: PLC0415

    settings = get_settings()
    if since is None:
        since = (datetime.utcnow() - timedelta(days=90)).date()

    conn = await asyncpg.connect(settings.database_url)
    try:
        rows = await conn.fetch(
            """
            SELECT
                i.inspection_id,
                i.inspection_date  AS date,
                i.result,
                i.bbl,
                ST_Y(i.geom)       AS lat,
                ST_X(i.geom)       AS lon
            FROM raw.rodent_inspections i
            JOIN raw.nta_boundaries b ON ST_Within(i.geom, b.geom)
            WHERE b.nta_id = $1
              AND i.inspection_date >= $2
              AND i.geom IS NOT NULL
            ORDER BY i.inspection_date DESC
            LIMIT 500
            """,
            nta_id,
            since,
        )
    finally:
        await conn.close()

    return [
        InspectionItem(
            inspection_id=r["inspection_id"],
            date=r["date"],
            result=r["result"],
            bbl=r["bbl"],
            lat=float(r["lat"]) if r["lat"] is not None else None,
            lon=float(r["lon"]) if r["lon"] is not None else None,
        )
        for r in rows
    ]
