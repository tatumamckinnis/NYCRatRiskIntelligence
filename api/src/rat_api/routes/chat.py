"""POST /chat — SSE streaming chat endpoint (T-41).

Request body: ``{"question": "...", "session_id": "<uuid-or-null>"}``

Streams ``data: <token>\\n\\n`` SSE frames; final frame is ``data: [DONE]\\n\\n``.
Sets ``X-Session-Id`` response header to the session UUID (new or existing).
"""

from __future__ import annotations

import logging
import uuid
from typing import AsyncIterator

import asyncpg
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rat_api.config import get_settings
from rat_api.obs.tracing import chain_span
from rat_api.rag.generator import generate_stream
from rat_api.rag.retriever import retrieve

log = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    session_id: uuid.UUID | None = None


async def _get_or_create_session(
    session_id: uuid.UUID | None,
    conn: asyncpg.Connection,
) -> uuid.UUID:
    if session_id is not None:
        # Verify it exists; if not, create a new one
        row = await conn.fetchrow(
            "SELECT session_id FROM app.chat_sessions WHERE session_id = $1", session_id
        )
        if row:
            return session_id

    new_id = uuid.uuid4()
    await conn.execute(
        "INSERT INTO app.chat_sessions (session_id) VALUES ($1)", new_id
    )
    return new_id


async def _sse_stream(
    question: str,
    session_id: uuid.UUID,
    db_url: str,
) -> AsyncIterator[str]:
    """Open a DB connection and stream SSE frames."""
    conn = await asyncpg.connect(db_url)
    try:
        with chain_span("chat_request") as span:
            span.set_attribute("session_id", str(session_id))
            span.set_attribute("question", question[:200])

            chunks = await retrieve(question, conn)
            async for token in generate_stream(
                question, chunks, session_id=session_id, conn=conn
            ):
                yield f"data: {token}\n\n"

        yield "data: [DONE]\n\n"
    except Exception as exc:  # noqa: BLE001
        log.error("Chat stream error: %s", exc)
        yield f"data: [ERROR] {exc}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        await conn.close()


@router.post("/chat")
async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
    """Stream a cited answer to *body.question* from the NYC Health Code corpus."""
    settings = get_settings()
    db_url = settings.database_url

    # Resolve session ID (create if needed)
    conn = await asyncpg.connect(db_url)
    try:
        session_id = await _get_or_create_session(body.session_id, conn)
    finally:
        await conn.close()

    response = StreamingResponse(
        _sse_stream(body.question, session_id, db_url),
        media_type="text/event-stream",
    )
    response.headers["X-Session-Id"] = str(session_id)
    response.headers["Cache-Control"] = "no-cache"
    return response
