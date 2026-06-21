"""GET /risk/* endpoints (T-20, T-28)."""

from __future__ import annotations

from datetime import date, timedelta

import asyncpg
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from rat_api.config import get_settings
from rat_api.ml.features import (
    current_iso_week,
    get_all_nta_features_for_week,
    get_nta_features,
)
from rat_api.ml.predict import predict_risk
from rat_api.models.risk import MapRiskItem, NtaRiskResponse, WeekForecast

router = APIRouter(prefix="/risk")


def _stub_forecast(current_week: date, risk_score: float) -> list[WeekForecast]:
    """Return a 12-week flat stub when TFT forecasts are unavailable."""
    return [
        WeekForecast(
            week=current_week + timedelta(weeks=i),
            risk_score=round(risk_score, 6),
            ci_low=max(0.0, round(risk_score - 0.05, 6)),
            ci_high=min(1.0, round(risk_score + 0.05, 6)),
        )
        for i in range(1, 13)
    ]


async def _get_tft_forecast(
    nta_id: str,
    as_of_week: date,
    conn: asyncpg.Connection,
) -> list[WeekForecast] | None:
    """Fetch TFT forecast rows from app.tft_forecasts. Returns None when absent."""
    try:
        rows = await conn.fetch(
            """
            SELECT forecast_week, p10, p50, p90
            FROM app.tft_forecasts
            WHERE nta_id = $1 AND as_of_week = $2
            ORDER BY forecast_week
            LIMIT 12
            """,
            nta_id,
            as_of_week,
        )
    except Exception:  # noqa: BLE001
        return None

    if not rows:
        return None

    return [
        WeekForecast(
            week=r["forecast_week"],
            risk_score=round(float(r["p50"]), 6),
            ci_low=round(max(0.0, float(r["p10"])), 6),
            ci_high=round(min(1.0, float(r["p90"])), 6),
        )
        for r in rows
    ]


@router.get("/nta/{nta_id}", response_model=NtaRiskResponse)
async def get_nta_risk(nta_id: str, request: Request) -> JSONResponse:
    """Return the current-week risk score and top factors for one NTA.

    Returns 503 if the current-week feature row is missing.

    ``forecast_12w`` uses TFT probabilistic forecasts (p10/p50/p90) when
    materialised by ``materialize_tft_forecasts.py``.  Falls back to a flat
    stub when TFT rows are absent; ``X-Forecast-Stub: true`` header signals
    this condition to clients.
    """
    settings = get_settings()
    model_bundle = getattr(request.app.state, "model_bundle", None)
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    week = current_iso_week()
    conn = await asyncpg.connect(settings.database_url)
    try:
        feature_row = await get_nta_features(nta_id, week, conn)
        if feature_row is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"No feature row for nta_id={nta_id!r} week={week}. "
                    "Ingest may be behind; try again later."
                ),
            )
        tft_forecast = await _get_tft_forecast(nta_id, week, conn)
    finally:
        await conn.close()

    result = predict_risk(
        model=model_bundle["model"],
        feature_row=feature_row,
        feature_cols=model_bundle["metadata"]["feature_cols"],
        decile_thresholds=request.app.state.decile_thresholds,
        model_version=model_bundle["version"],
    )

    is_stub = tft_forecast is None
    forecast = tft_forecast or _stub_forecast(week, result.risk_score)

    body = NtaRiskResponse(
        nta_id=nta_id,
        current_week=week,
        risk_score=result.risk_score,
        risk_decile=result.risk_decile,
        top_factors=result.top_factors,
        model_version=result.model_version,
        forecast_12w=forecast,
    )
    headers = {"X-Forecast-Stub": "true"} if is_stub else {}
    return JSONResponse(content=body.model_dump(mode="json"), headers=headers)


@router.get("/map", response_model=list[MapRiskItem])
async def get_risk_map(
    request: Request,
    week: date = Query(default=None, description="ISO week start (Monday). Defaults to current week."),
) -> list[MapRiskItem]:
    """Return risk scores for all NTAs for a given week.

    Reads from ``app.risk_predictions`` when available (pre-materialised by
    ``materialize_predictions.py``).  Falls back to live inference across all
    NTAs if the cache table is empty for the requested week.
    """
    settings = get_settings()
    model_bundle = getattr(request.app.state, "model_bundle", None)
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    if week is None:
        week = current_iso_week()

    conn = await asyncpg.connect(settings.database_url)
    try:
        # Fetch NTA name + centroid for all boundaries once
        boundary_rows = await conn.fetch(
            """
            SELECT nta_id, nta_name,
                   ST_Y(ST_Centroid(geom)) AS centroid_lat,
                   ST_X(ST_Centroid(geom)) AS centroid_lon
            FROM raw.nta_boundaries
            """
        )
        boundaries = {r["nta_id"]: r for r in boundary_rows}

        # Try the materialised cache first
        cached = await conn.fetch(
            """
            SELECT nta_id, risk_score, risk_decile
            FROM app.risk_predictions
            WHERE predicted_for_week = $1
              AND model_version = $2
            ORDER BY nta_id
            """,
            week,
            model_bundle["version"],
        )
        if cached:
            return [
                MapRiskItem(
                    nta_id=r["nta_id"],
                    risk_score=float(r["risk_score"]),
                    risk_decile=int(r["risk_decile"]),
                    nta_name=boundaries.get(r["nta_id"], {}).get("nta_name"),
                    centroid_lat=boundaries.get(r["nta_id"], {}).get("centroid_lat"),
                    centroid_lon=boundaries.get(r["nta_id"], {}).get("centroid_lon"),
                )
                for r in cached
            ]

        # Fallback: live inference for all NTAs
        all_features = await get_all_nta_features_for_week(week, conn)
    finally:
        await conn.close()

    if not all_features:
        return []

    items: list[MapRiskItem] = []
    for row in all_features:
        result = predict_risk(
            model=model_bundle["model"],
            feature_row=row,
            feature_cols=model_bundle["metadata"]["feature_cols"],
            decile_thresholds=request.app.state.decile_thresholds,
            model_version=model_bundle["version"],
        )
        b = boundaries.get(row["nta_id"], {})
        items.append(
            MapRiskItem(
                nta_id=row["nta_id"],
                risk_score=result.risk_score,
                risk_decile=result.risk_decile,
                nta_name=b.get("nta_name"),
                centroid_lat=b.get("centroid_lat"),
                centroid_lon=b.get("centroid_lon"),
            )
        )
    return sorted(items, key=lambda x: x.nta_id)
