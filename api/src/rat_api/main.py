"""FastAPI application entry point (T-18)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import numpy as np
import asyncpg
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rat_api.config import get_settings
from rat_api.ml.loader import load_models
from rat_api.ml.features import get_all_nta_features_for_week, current_iso_week
from rat_api.ml.predict import predict_risk
from rat_api.routes import health, inspections, narrative, risk


def _compute_decile_thresholds(scores: list[float]) -> list[float]:
    """Return 10 thresholds (10th … 100th percentile) from a score list."""
    if not scores:
        return [i / 10 for i in range(1, 11)]
    arr = np.array(scores)
    return [float(np.percentile(arr, p)) for p in range(10, 101, 10)]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load models and pre-compute decile thresholds at startup."""
    settings = get_settings()

    # Sentry (no-op if DSN is empty)
    if settings.sentry_dsn:
        import sentry_sdk  # noqa: PLC0415
        sentry_sdk.init(dsn=settings.sentry_dsn, traces_sample_rate=0.1)

    # Load model artifacts
    try:
        bundle = load_models(settings.model_artifacts_dir, settings.model_name)
        app.state.model_bundle = bundle
    except (FileNotFoundError, KeyError) as exc:
        # Start in degraded mode — /health will return 503
        app.state.model_bundle = None
        app.state.startup_error = str(exc)
        app.state.decile_thresholds = [i / 10 for i in range(1, 11)]
        yield
        return

    # Pre-compute decile thresholds from the current week's score distribution
    try:
        conn = await asyncpg.connect(settings.database_url)
        try:
            week = current_iso_week()
            all_features = await get_all_nta_features_for_week(week, conn)
        finally:
            await conn.close()

        scores = []
        for row in all_features:
            try:
                result = predict_risk(
                    model=bundle["model"],
                    feature_row=row,
                    feature_cols=bundle["metadata"]["feature_cols"],
                    decile_thresholds=[i / 10 for i in range(1, 11)],
                    model_version=bundle["version"],
                )
                scores.append(result.risk_score)
            except Exception:  # noqa: BLE001
                continue
        app.state.decile_thresholds = _compute_decile_thresholds(scores)
    except Exception:  # noqa: BLE001
        app.state.decile_thresholds = [i / 10 for i in range(1, 11)]

    yield  # app is running


def create_app() -> FastAPI:
    app = FastAPI(
        title="NYC Rat Risk Intelligence API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(risk.router)
    app.include_router(inspections.router)
    app.include_router(narrative.router)

    return app


app = create_app()
