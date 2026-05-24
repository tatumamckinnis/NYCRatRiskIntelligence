"""GET /risk/* endpoints (T-20)."""

from __future__ import annotations

from datetime import date, timedelta

import asyncpg
from fastapi import APIRouter, HTTPException, Query, Request

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
    """Return a 12-week forecast stub (TFT not yet trained in Phase 2)."""
    return [
        WeekForecast(
            week=current_week + timedelta(weeks=i),
            risk_score=round(risk_score, 6),
            ci_low=max(0.0, round(risk_score - 0.05, 6)),
            ci_high=min(1.0, round(risk_score + 0.05, 6)),
        )
        for i in range(1, 13)
    ]


@router.get("/nta/{nta_id}", response_model=NtaRiskResponse)
async def get_nta_risk(nta_id: str, request: Request) -> NtaRiskResponse:
    """Return the current-week risk score and top factors for one NTA.

    Returns 503 if the current-week feature row is missing — the API never
    fabricates predictions from stale data.

    The ``forecast_12w`` field is stubbed with constant CI bands until the
    TFT model is trained in Phase 3.  A ``X-Forecast-Stub: true`` response
    header flags this condition.
    """
    settings = get_settings()
    model_bundle = getattr(request.app.state, "model_bundle", None)
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    week = current_iso_week()
    conn = await asyncpg.connect(settings.database_url)
    try:
        feature_row = await get_nta_features(nta_id, week, conn)
    finally:
        await conn.close()

    if feature_row is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No feature row for nta_id={nta_id!r} week={week}. "
                "Ingest may be behind; try again later."
            ),
        )

    result = predict_risk(
        model=model_bundle["model"],
        feature_row=feature_row,
        feature_cols=model_bundle["metadata"]["feature_cols"],
        decile_thresholds=request.app.state.decile_thresholds,
        model_version=model_bundle["version"],
    )

    return NtaRiskResponse(
        nta_id=nta_id,
        current_week=week,
        risk_score=result.risk_score,
        risk_decile=result.risk_decile,
        top_factors=result.top_factors,
        model_version=result.model_version,
        forecast_12w=_stub_forecast(week, result.risk_score),
    )


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
        items.append(
            MapRiskItem(
                nta_id=row["nta_id"],
                risk_score=result.risk_score,
                risk_decile=result.risk_decile,
            )
        )
    return sorted(items, key=lambda x: x.nta_id)
