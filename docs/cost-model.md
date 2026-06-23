# Cost model

**Target: $0/month recurring at launch traffic.**

---

## Current infrastructure costs (launch)

| Line item | Monthly cost | Limit / notes |
|---|---|---|
| Supabase Free | $0 | 500 MB storage; current DB ≈ 280 MB |
| Render free tier | $0 | 512 MB RAM; 15-min spin-down; cold start ~30s |
| Vercel Hobby | $0 | 100 GB bandwidth/month |
| Custom domain / DNS | $0 | Using `*.vercel.app` + `*.onrender.com` |
| CartoDB Positron basemap | $0 | Public tile server; no API key required |
| BGE-M3 embeddings | $0 | Self-hosted in API container (disabled in prod) |
| BGE Reranker v2-M3 | $0 | Self-hosted in API container (disabled in prod) |
| Groq (LLM generation) | $0 | Free tier: 30 req/min, 6k tokens/min |
| Sentry free tier | $0 | 5K errors/month |
| **Total recurring** | **$0/month** | |

**One-time development spend:** ≤$5 (Anthropic API credit for ~500 dev/eval queries)

---

## Traffic assumptions (launch)

| Signal | Estimate |
|---|---|
| Daily active users | 5–20 (demo / portfolio traffic) |
| `/risk/map` calls/day | 50–200 |
| `/chat` calls/day | 10–50 |
| Map tile loads/day | 500–2,000 |

At this scale every free tier has comfortable headroom.

---

## 10× traffic — what breaks first

At 10× (~500 daily users, 2k map calls/day, 500 chat calls/day):

| Bottleneck | Impact | Fix |
|---|---|---|
| Render 512 MB RAM | Groq streaming + asyncpg pool may OOM if concurrent requests spike | Upgrade to Render Starter ($7/mo, 512 MB guaranteed + no spin-down) |
| Supabase 500 MB storage | Risk predictions table (~40 MB), chunked docs (~60 MB) leave ~400 MB for growth | Stay on free tier; prune old traces |
| Groq rate limit (30 req/min) | Chat queue backpressure at >30 concurrent chat sessions | Add per-user rate limit; queue with asyncio.Semaphore |
| Render cold starts | No change — Render Starter has no spin-down | Covered by Starter upgrade |

**Estimated cost at 10×:** $7/month (Render Starter only)

---

## 100× traffic — paid tier stack

At 100× (~5k daily users):

| Service | Upgrade | Estimated cost |
|---|---|---|
| Render | Standard instance (2 GB RAM, always-on) | $25/month |
| Supabase | Pro ($25/mo, 8 GB storage, PITR) | $25/month |
| Vercel | Pro ($20/mo, 1 TB bandwidth) | $20/month |
| LLM (Groq or Anthropic) | Pay-as-you-go; ~500 chat calls/day × 2k tokens = ~1B tokens/month | ~$5–15/month |
| **Total** | | **~$75–85/month** |

At 100×, the BGE-M3 + Reranker can be re-enabled (Standard instance has 2 GB RAM), which improves RAG quality materially (hybrid retrieval vs. BM25-only).

---

## Per-query cost targets

| Query type | Target | Current actual |
|---|---|---|
| `/risk/map` (cached) | ≤$0.000001 | ~$0 (DB read only) |
| `/risk/nta/{id}` (model inference) | ≤$0.001 | ~$0 (CPU inference, <10ms) |
| `/chat` RAG query | ≤$0.005 | ~$0 (Groq free tier) |

---

## Credit burn guardrails

Every LLM call logs `llm.usd_cost` computed from token counts × hardcoded price table in `api/src/rat_api/config.py`. A middleware check warns (log + Sentry alert) if cumulative spend in the current calendar day exceeds $1. No hard-stop is implemented (to avoid breaking live demos), but the alert provides time to intervene.
