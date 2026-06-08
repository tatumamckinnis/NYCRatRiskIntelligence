"""Tests for the /chat SSE endpoint (T-43)."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


async def _fake_generate_stream(query, chunks, *, session_id, conn):
    yield "Hello "
    yield "world."


@pytest.mark.asyncio
async def test_chat_returns_event_stream_content_type():
    from rat_api.main import app

    with (
        patch("rat_api.routes.chat._get_or_create_session", new_callable=AsyncMock, return_value=uuid.uuid4()),
        patch("rat_api.routes.chat.retrieve", new_callable=AsyncMock, return_value=[]),
        patch("rat_api.routes.chat.generate_stream", side_effect=_fake_generate_stream),
        patch("asyncpg.connect", new_callable=AsyncMock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            async with client.stream("POST", "/chat", json={"question": "What is §151.02?"}) as resp:
                assert resp.status_code == 200
                assert "text/event-stream" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_chat_returns_done_frame():
    from rat_api.main import app

    chunks_received = []

    with (
        patch("rat_api.routes.chat._get_or_create_session", new_callable=AsyncMock, return_value=uuid.uuid4()),
        patch("rat_api.routes.chat.retrieve", new_callable=AsyncMock, return_value=[]),
        patch("rat_api.routes.chat.generate_stream", side_effect=_fake_generate_stream),
        patch("asyncpg.connect", new_callable=AsyncMock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            async with client.stream("POST", "/chat", json={"question": "test"}) as resp:
                async for line in resp.aiter_lines():
                    chunks_received.append(line)

    full_text = "\n".join(chunks_received)
    assert "data: [DONE]" in full_text, f"No [DONE] frame in: {full_text!r}"


@pytest.mark.asyncio
async def test_chat_sets_session_id_header():
    from rat_api.main import app

    fixed_session = uuid.uuid4()

    with (
        patch("rat_api.routes.chat._get_or_create_session", new_callable=AsyncMock, return_value=fixed_session),
        patch("rat_api.routes.chat.retrieve", new_callable=AsyncMock, return_value=[]),
        patch("rat_api.routes.chat.generate_stream", side_effect=_fake_generate_stream),
        patch("asyncpg.connect", new_callable=AsyncMock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            async with client.stream("POST", "/chat", json={"question": "test"}) as resp:
                # Consume stream
                async for _ in resp.aiter_lines():
                    pass
                assert "x-session-id" in resp.headers
                assert resp.headers["x-session-id"] == str(fixed_session)
