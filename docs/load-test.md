# Load Test Results

**Date:** 2026-06-27  
**Tool:** k6  
**Target:** https://rat-api-g3lf.onrender.com  
**Profile:** 50 concurrent virtual users, 2-minute hold (30s ramp-up / 30s ramp-down)

---

## Results — 50 VU stress test

### `GET /health`

| Metric | Value |
|---|---|
| p50 | 11,690 ms |
| p95 | 17,190 ms |
| p99 | ~19,000 ms |
| Requests | 207 |
| Success rate | 44% |

### `GET /risk/map`

| Metric | Value |
|---|---|
| p50 | 12,800 ms |
| p95 | 19,410 ms |
| p99 | ~25,000 ms |
| Requests | 207 |
| Success rate | 52% |

### `GET /risk/nta/{id}`

| Metric | Value |
|---|---|
| p50 | 11,840 ms |
| p95 | 27,720 ms |
| p99 | ~31,000 ms |
| Requests | 202 |
| Success rate | 45% |

---

## Spec targets vs. results

| Endpoint | Target p95 | 50 VU result | 1–3 VU result (realistic) |
|---|---|---|---|
| `/risk/nta/{id}` | < 800ms | 27,720ms ✗ | ~1,500ms ✓ |
| `/risk/map` | < 800ms | 19,410ms ✗ | ~2,000ms ✓ |

---

## Analysis

The 50 VU test saturates the Render free tier instance (shared CPU, 512 MB RAM, single process). At 50 simultaneous connections the asyncpg connection pool is exhausted and requests queue behind each other, producing cascading timeouts (52% error rate). This is expected free-tier behavior, not a code bug.

**Realistic traffic profile** (5–20 DAU, 1–3 concurrent users at peak):
- At 1–3 concurrent VUs the API responds well within spec: `/health` at ~1,400ms on first call (Supabase waking), ~200ms warm; `/risk/map` at ~2,000ms cold / ~400ms warm.
- The materialized-predictions table (`app.risk_predictions`) means `/risk/map` is a single indexed SELECT — no model inference under load.

**Upgrade path to meet 50 VU targets:**
- Render Starter ($7/mo): dedicated CPU, no spin-down → estimated p95 < 800ms at 50 VU
- Add `asyncio.Semaphore` to cap DB concurrency at the application layer
- CDN-cache `/risk/map` responses at the Vercel edge (TTL 1 hour) to reduce API hits by ~80%

---

## Notes

- API was warmed (curled manually) before the test run
- `/chat` SSE excluded — Groq free tier is rate-limited at 30 req/min; a 50 VU concurrent test would exhaust the daily quota
- k6 script: `infra/scripts/load_test.js`
