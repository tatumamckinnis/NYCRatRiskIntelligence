"""Unit tests for GET /health (T-22).

All DB and model state is patched so these tests run without a live database.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from rat_api.routes.health import router

# ---------------------------------------------------------------------------
# Minimal app fixture — mounts only the health router so tests are isolated
# ---------------------------------------------------------------------------


def _make_app(*, model_loaded: bool) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    bundle = {"version": "20240101_120000"} if model_loaded else None
    app.state.model_bundle = bundle
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_ok_when_db_and_model_ready():
    app = _make_app(model_loaded=True)

    async def _fake_connect(*args, **kwargs):
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=1)
        conn.close = AsyncMock()
        return conn

    with patch("rat_api.routes.health.asyncpg.connect", side_effect=_fake_connect):
        with patch("rat_api.routes.health.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(database_url="postgresql://fake/db")
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "20240101_120000"
    assert "db_latency_ms" in body
    assert "git_sha" in body


@pytest.mark.asyncio
async def test_health_degraded_when_model_not_loaded():
    app = _make_app(model_loaded=False)

    async def _fake_connect(*args, **kwargs):
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=1)
        conn.close = AsyncMock()
        return conn

    with patch("rat_api.routes.health.asyncpg.connect", side_effect=_fake_connect):
        with patch("rat_api.routes.health.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(database_url="postgresql://fake/db")
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/health")

    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["model_version"] == "not_loaded"


@pytest.mark.asyncio
async def test_health_degraded_when_db_unreachable():
    app = _make_app(model_loaded=True)

    with patch(
        "rat_api.routes.health.asyncpg.connect",
        side_effect=ConnectionRefusedError("no db"),
    ):
        with patch("rat_api.routes.health.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(database_url="postgresql://fake/db")
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/health")

    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
