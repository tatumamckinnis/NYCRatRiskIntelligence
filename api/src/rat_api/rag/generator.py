"""Streaming LLM generation for the /chat endpoint (T-41).

Uses ``anthropic.AsyncAnthropic().messages.stream()`` with ``claude-haiku-4-5``.
Wraps the call in an ``llm_span`` that records token counts and USD cost.
Persists the completed message pair to ``app.chat_messages``.
"""

from __future__ import annotations

import logging
import uuid
from typing import AsyncIterator, TYPE_CHECKING

import asyncpg

from rat_api.config import get_settings
from rat_api.obs.tracing import llm_span
from rat_api.rag.prompts import CHAT_SYSTEM_PROMPT

if TYPE_CHECKING:
    from rat_api.rag.retriever import RetrievedChunk

log = logging.getLogger(__name__)

_MODEL = "claude-haiku-4-5"
_MAX_CONTEXT_CHARS = 4000  # per-chunk truncation for context window safety


def _build_context(chunks: "list[RetrievedChunk]") -> str:
    """Format retrieved chunks as numbered context blocks."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        text = c.content_with_prefix[:_MAX_CONTEXT_CHARS]
        parts.append(f"[{i}] {c.citation} ({c.authority})\n{text}")
    return "\n\n".join(parts)


async def generate_stream(
    query: str,
    chunks: "list[RetrievedChunk]",
    *,
    session_id: uuid.UUID,
    conn: asyncpg.Connection,
) -> AsyncIterator[str]:
    """Stream LLM tokens for *query* grounded in *chunks*.

    Yields token strings as they arrive.  When the stream is complete,
    persists the user message and assistant response to ``app.chat_messages``.

    Args:
        query:      User question.
        chunks:     Retrieved chunks to use as context.
        session_id: Chat session UUID.
        conn:       Active asyncpg connection for message persistence.
    """
    import anthropic  # noqa: PLC0415

    settings = get_settings()
    if not settings.anthropic_api_key:
        yield "Error: ANTHROPIC_API_KEY is not configured."
        return

    context = _build_context(chunks)
    user_content = f"Context:\n{context}\n\nQuestion: {query}"

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    full_response = []
    prompt_tokens = 0
    completion_tokens = 0

    with llm_span("chat_generate") as span:
        span.set_attribute("llm.model_name", _MODEL)
        span.set_attribute("llm.provider", "anthropic")
        span.set_attribute("llm.input_messages", f"system:{CHAT_SYSTEM_PROMPT[:200]}…")

        try:
            async with client.messages.stream(
                model=_MODEL,
                max_tokens=1024,
                system=CHAT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            ) as stream:
                async for text in stream.text_stream:
                    full_response.append(text)
                    yield text

                # Final message for token counts
                final_msg = await stream.get_final_message()
                prompt_tokens = final_msg.usage.input_tokens
                completion_tokens = final_msg.usage.output_tokens

        except Exception as exc:  # noqa: BLE001
            log.error("LLM stream error: %s", exc)
            yield f"\n[Error: {exc}]"
            return

        # Cost calculation
        price_in = settings.llm_price_per_1m_input.get(_MODEL, 0.80)
        price_out = settings.llm_price_per_1m_output.get(_MODEL, 4.00)
        cost_usd = (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000

        span.set_attribute("llm.token_count.prompt", prompt_tokens)
        span.set_attribute("llm.token_count.completion", completion_tokens)
        span.set_attribute("llm.token_count.total", prompt_tokens + completion_tokens)
        span.set_attribute("llm.usd_cost", cost_usd)

    # Persist to app.chat_messages
    assistant_text = "".join(full_response)
    await _persist_messages(
        conn=conn,
        session_id=session_id,
        user_content=query,
        assistant_content=assistant_text,
        chunks=chunks,
        latency_ms=None,
        cost_usd=cost_usd,
    )


async def _persist_messages(
    *,
    conn: asyncpg.Connection,
    session_id: uuid.UUID,
    user_content: str,
    assistant_content: str,
    chunks: "list[RetrievedChunk]",
    latency_ms: int | None,
    cost_usd: float,
) -> None:
    """Write user + assistant messages to ``app.chat_messages``."""
    import json  # noqa: PLC0415

    retrieved_json = json.dumps(
        [{"chunk_id": c.chunk_id, "score": c.score} for c in chunks]
    )
    sql = """
        INSERT INTO app.chat_messages
            (session_id, role, content, retrieved_chunks, latency_ms, cost_usd)
        VALUES ($1, $2, $3, $4, $5, $6)
    """
    try:
        await conn.execute(sql, session_id, "user", user_content, None, None, None)
        await conn.execute(
            sql,
            session_id,
            "assistant",
            assistant_content,
            retrieved_json,
            latency_ms,
            cost_usd,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to persist chat messages: %s", exc)
