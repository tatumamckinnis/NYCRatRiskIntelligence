"""Unit tests for GET /risk/* endpoints (T-22).

DB calls are patched via asyncpg mocks; model inference uses a real
IsotonicCalibratedClassifier-compatible stub so the predict pipeline
runs end-to-end without SHAP.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from rat_api.routes.risk import router

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "complaints_count",
    "complaints_lag_1w",
    "complaints_lag_4w",
    "complaints_lag_12w",
    "weather_tavg_c",
    "units_total",
]

FAKE_WEEK = date(2024, 5, 6)  # Monday


class _FakeModel:
    def predict_proba(self, X):  # noqa: ANN001, N803
        return np.array([[0.3, 0.7]] * len(X))


def _make_bundle() -> dict:
    return {
        "model": _FakeModel(),
        "metadata": {"feature_cols": FEATURE_COLS},
        "model_name": "catboost",
        "version": "20240101_120000",
    }


def _make_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.model_bundle = _make_bundle()
    app.state.decile_thresholds = [i / 10 for i in range(1, 11)]
    return app


def _feature_row(nta_id: str = "MN2501") -> dict:
    return {col: 1.0 for col in FEATURE_COLS} | {"nta_id": nta_id}


# ---------------------------------------------------------------------------
# GET /risk/nta/{nta_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_risk_nta_returns_200():
    app = _make_app()

    async def _fake_connect(*args, **kwargs):
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=_feature_row())
        conn.close = AsyncMock()
        return conn

    with patch("rat_api.routes.risk.asyncpg.connect", side_effect=_fake_connect):
        with patch("rat_api.routes.risk.get_settings") as ms:
            ms.return_value = MagicMock(database_url="postgresql://fake/db")
            with patch("rat_api.routes.risk.current_iso_week", return_value=FAKE_WEEK):
                with patch(
                    "rat_api.routes.risk.get_nta_features",
                    new=AsyncMock(return_value=_feature_row()),
                ):
                    async with AsyncClient(
                        transport=ASGITransport(app=app), base_url="http://test"
                    ) as client:
                        resp = await client.get("/risk/nta/MN2501")

    assert resp.status_code == 200
    body = resp.json()
    assert body["nta_id"] == "MN2501"
    assert 0.0 <= body["risk_score"] <= 1.0
    assert 1 <= body["risk_decile"] <= 10
    assert body["model_version"] == "20240101_120000"
    assert len(body["forecast_12w"]) == 12


@pytest.mark.asyncio
async def test_risk_nta_503_when_feature_row_missing():
    app = _make_app()

    with patch("rat_api.routes.risk.asyncpg.connect"):
        with patch("rat_api.routes.risk.get_settings") as ms:
            ms.return_value = MagicMock(database_url="postgresql://fake/db")
            with patch("rat_api.routes.risk.current_iso_week", return_value=FAKE_WEEK):
                with patch(
                    "rat_api.routes.risk.get_nta_features",
                    new=AsyncMock(return_value=None),
                ):
                    async with AsyncClient(
                        transport=ASGITransport(app=app), base_url="http://test"
                    ) as client:
                        resp = await client.get("/risk/nta/UNKNOWN")

    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_risk_nta_503_when_model_not_loaded():
    app = _make_app()
    app.state.model_bundle = None

    with patch("rat_api.routes.risk.get_settings") as ms:
        ms.return_value = MagicMock(database_url="postgresql://fake/db")
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/risk/nta/MN2501")

    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# GET /risk/map
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_risk_map_returns_list():
    app = _make_app()
    rows = [_feature_row(f"BK{i:04d}") for i in range(5)]

    async def _fake_connect(*args, **kwargs):
        conn = AsyncMock()
        # cached table is empty → triggers live inference path
        conn.fetch = AsyncMock(side_effect=[[], rows])
        conn.close = AsyncMock()
        return conn

    with patch("rat_api.routes.risk.asyncpg.connect", side_effect=_fake_connect):
        with patch("rat_api.routes.risk.get_settings") as ms:
            ms.return_value = MagicMock(database_url="postgresql://fake/db")
            with patch("rat_api.routes.risk.current_iso_week", return_value=FAKE_WEEK):
                with patch(
                    "rat_api.routes.risk.get_all_nta_features_for_week",
                    new=AsyncMock(return_value=rows),
                ):
                    async with AsyncClient(
                        transport=ASGITransport(app=app), base_url="http://test"
                    ) as client:
                        resp = await client.get("/risk/map")

    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    for item in body:
        assert "nta_id" in item
        assert 0.0 <= item["risk_score"] <= 1.0
        assert 1 <= item["risk_decile"] <= 10


@pytest.mark.asyncio
async def test_risk_map_empty_when_no_features():
    app = _make_app()

    with patch("rat_api.routes.risk.asyncpg.connect"):
        with patch("rat_api.routes.risk.get_settings") as ms:
            ms.return_value = MagicMock(database_url="postgresql://fake/db")
            with patch("rat_api.routes.risk.current_iso_week", return_value=FAKE_WEEK):
                with patch(
                    "rat_api.routes.risk.get_all_nta_features_for_week",
                    new=AsyncMock(return_value=[]),
                ):
                    async with AsyncClient(
                        transport=ASGITransport(app=app), base_url="http://test"
                    ) as client:
                        resp = await client.get("/risk/map")

    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_risk_map_503_when_model_not_loaded():
    app = _make_app()
    app.state.model_bundle = None

    with patch("rat_api.routes.risk.get_settings") as ms:
        ms.return_value = MagicMock(database_url="postgresql://fake/db")
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/risk/map")

    assert resp.status_code == 503
