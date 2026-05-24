"""GET /health endpoint (T-20)."""

from __future__ import annotations

import subprocess
import time

import asyncpg
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from rat_api.config import get_settings
from rat_api.models.risk import HealthResponse

router = APIRouter()


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> JSONResponse:
    """Return 200 when DB is reachable and models are loaded; 503 otherwise."""
    settings = get_settings()
    model_state = getattr(request.app.state, "model_bundle", None)
    model_version = (
        model_state.get("version", "unknown") if model_state else "not_loaded"
    )

    # DB ping
    t0 = time.monotonic()
    try:
        conn = await asyncpg.connect(settings.database_url)
        await conn.fetchval("SELECT 1")
        await conn.close()
        db_ok = True
    except Exception:  # noqa: BLE001
        db_ok = False
    db_latency_ms = int((time.monotonic() - t0) * 1000)

    status = "ok" if (db_ok and model_state is not None) else "degraded"
    body = HealthResponse(
        status=status,
        model_version=model_version,
        db_latency_ms=db_latency_ms,
        git_sha=_git_sha(),
    )

    return JSONResponse(
        content=body.model_dump(),
        status_code=200 if status == "ok" else 503,
    )
