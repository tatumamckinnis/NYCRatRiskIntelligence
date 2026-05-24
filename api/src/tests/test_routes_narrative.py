"""Unit tests for GET /risk/nta/{nta_id}/narrative (T-29).

All external calls (DB, Voyage AI, Cohere, Claude) are mocked.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from rat_api.routes.narrative import router


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "complaints_count",
    "complaints_lag_1w",
    "weather_tavg_c",
    "units_total",
]
FAKE_WEEK = date(2024, 5, 6)


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


def _make_app(*, model_loaded: bool = True) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.model_bundle = _make_bundle() if model_loaded else None
    app.state.decile_thresholds = [i / 10 for i in range(1, 11)]
    return app


def _feature_row() -> dict:
    return {col: 1.0 for col in FEATURE_COLS} | {"nta_id": "MN2501"}


def _fake_settings(**kwargs):
    defaults = dict(
        database_url="postgresql://fake/db",
        anthropic_api_key="sk-test",
        voyageai_api_key="",
        cohere_api_key="",
    )
    defaults.update(kwargs)
    return MagicMock(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_narrative_returns_200():
    app = _make_app()

    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value=None)
    conn.fetch = AsyncMock(return_value=[])
    conn.close = AsyncMock()

    with patch("rat_api.routes.narrative.get_settings", return_value=_fake_settings()):
        with patch("rat_api.routes.narrative.asyncpg.connect", return_value=conn):
            with patch(
                "rat_api.routes.narrative.get_nta_features",
                new=AsyncMock(return_value=_feature_row()),
            ):
                with patch(
                    "rat_api.routes.narrative.retrieve",
                    new=AsyncMock(return_value=[]),
                ):
                    with patch(
                        "rat_api.routes.narrative._call_claude",
                        return_value="Rodent risk is elevated this week.",
                    ):
                        with patch(
                            "rat_api.routes.narrative._write_cache",
                            new=AsyncMock(),
                        ):
                            with patch(
                                "rat_api.routes.narrative.current_iso_week",
                                return_value=FAKE_WEEK,
                            ):
                                async with AsyncClient(
                                    transport=ASGITransport(app=app),
                                    base_url="http://test",
                                ) as client:
                                    resp = await client.get("/risk/nta/MN2501/narrative")

    assert resp.status_code == 200
    body = resp.json()
    assert body["nta_id"] == "MN2501"
    assert body["narrative"] == "Rodent risk is elevated this week."
    assert 0.0 <= body["risk_score"] <= 1.0
    assert 1 <= body["risk_decile"] <= 10


@pytest.mark.asyncio
async def test_narrative_503_when_model_not_loaded():
    app = _make_app(model_loaded=False)

    with patch("rat_api.routes.narrative.get_settings", return_value=_fake_settings()):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/risk/nta/MN2501/narrative")

    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_narrative_503_when_no_anthropic_key():
    app = _make_app()

    with patch(
        "rat_api.routes.narrative.get_settings",
        return_value=_fake_settings(anthropic_api_key=""),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/risk/nta/MN2501/narrative")

    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_narrative_503_when_feature_row_missing():
    app = _make_app()

    conn = AsyncMock()
    conn.close = AsyncMock()

    with patch("rat_api.routes.narrative.get_settings", return_value=_fake_settings()):
        with patch("rat_api.routes.narrative.asyncpg.connect", return_value=conn):
            with patch(
                "rat_api.routes.narrative.get_nta_features",
                new=AsyncMock(return_value=None),
            ):
                with patch(
                    "rat_api.routes.narrative.current_iso_week",
                    return_value=FAKE_WEEK,
                ):
                    async with AsyncClient(
                        transport=ASGITransport(app=app), base_url="http://test"
                    ) as client:
                        resp = await client.get("/risk/nta/UNKNOWN/narrative")

    assert resp.status_code == 503
