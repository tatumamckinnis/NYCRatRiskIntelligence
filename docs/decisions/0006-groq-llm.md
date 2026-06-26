# ADR-0006 — Groq instead of Claude Haiku for LLM generation and eval judge

**Status**: Accepted
**Date**: 2026-06-10
**Deciders**: project owner

---

## Context

SPEC §2 locked Claude Haiku 4.5 as the generation LLM for `/chat` and Claude Sonnet 4.5 as the eval judge in `evals/src/judge.py`. During Phase 4 implementation two blockers surfaced:

1. **Billing setup required**: the Anthropic API requires a paid account with a credit card on file even for the $5 credit tier. Setting up billing for a portfolio demo with uncertain traffic exposes the project to unexpected charges if traffic spikes or the credit runs out.

2. **Credit exhaustion for always-on demo**: the $5 credit covers roughly 500 generation queries at Claude Haiku pricing (~$0.01 per typical RAG call). A publicly linked demo with no rate limiting could exhaust this in hours if the project gets shared. There is no free tier equivalent for Anthropic.

Groq's free tier provides:
- 30 req/min, 6,000 tokens/min, 500,000 tokens/day (as of 2026-06)
- No credit card required — API key is issued instantly
- `llama-3.3-70b-versatile` (generation) and `llama-3.1-8b-instant` (query rewriting) are available on the free tier
- OpenAI-compatible API — compatible with `litellm` routing

---

## Decision

Use **Groq free tier** as the sole LLM provider for production:
- Query rewriting: `groq/llama-3.1-8b-instant` (fast, small, no cost)
- `/chat` generation: `groq/llama-3.3-70b-versatile` (highest-quality free model)
- Eval faithfulness judge: `groq/llama-3.3-70b-versatile` (replaces Claude Sonnet 4.5)

`ANTHROPIC_API_KEY` and `OPENAI_API_KEY` are retained in `.env.example` as legacy entries for development use, but are not required for any production code path.

---

## Rationale

| Option | Pros | Cons |
|---|---|---|
| **Groq free tier (chosen)** | $0 forever; no billing risk; instant signup; OpenAI-compatible | Rate limited at 30 req/min; Llama quality slightly below Claude Haiku |
| Claude Haiku 4.5 | Spec-compliant; high quality; Anthropic native | Requires paid account; $5 credit finite; billing exposure |
| Claude Haiku via litellm | Same routing layer as Groq | Same billing issues as above |
| OpenAI GPT-4o-mini | Free trial available | Trial expires; no permanent free tier |

The output quality difference is acceptable for this use case: both models receive the same retrieved chunks in context, so factual grounding comes from retrieval, not the model's parametric knowledge. Citation formatting quality is comparable in manual testing.

**Why:** Budget is $0/month recurring. The project must be publicly demoable indefinitely without manual intervention.
**How to apply:** All `litellm.completion()` calls use `model="groq/..."` and `api_key=settings.groq_api_key`. Swapping to Claude requires only changing the model string and setting `ANTHROPIC_API_KEY` — no other code changes.

---

## Consequences

- `GROQ_API_KEY` is the only required LLM key in `.env.example` and `render.yaml`.
- The eval faithfulness judge (`evals/src/judge.py`) scores with Llama 3.3 70B, not Claude Sonnet. Judge quality may be slightly lower on edge cases; calibrate thresholds accordingly.
- Groq rate limit (30 req/min) becomes a bottleneck at >30 concurrent chat users. Mitigation: `asyncio.Semaphore(10)` on the generation call; document in the runbook.
- If Claude is ever re-enabled, update `model` in `generator.py`, `retriever.py`, and `judge.py`, and restore `ANTHROPIC_API_KEY` to required env vars.
- `llm.usd_cost` in trace spans is logged as `0.0` for all Groq calls (free tier). Cost tracking remains wired up for easy re-activation with paid providers.
