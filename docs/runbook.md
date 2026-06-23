# Runbook — NYC Rat Risk Intelligence

This document covers day-to-day operations for the deployed stack.

---

## Services at a glance

| Service | URL | Provider | SLA |
|---|---|---|---|
| API | https://rat-api-g3lf.onrender.com | Render free | No SLA; ~99% during business hours |
| Frontend | https://web-beige-three-56.vercel.app | Vercel Hobby | 99.99% (Vercel SLA) |
| Database | (Supabase project URL in env) | Supabase Free | 99.9% (Supabase SLA) |

---

## Health checks

```bash
# API health (expect {"status": "ok", "model_loaded": true})
curl https://rat-api-g3lf.onrender.com/health

# Map data endpoint (expect array of NTA risk items)
curl "https://rat-api-g3lf.onrender.com/risk/map?week=2026-05-11" | jq length
```

If `model_loaded` is `false`, the API is in degraded mode (503 on all `/risk/*` routes). Check the **Degraded mode** section below.

---

## Deployments

**API (Render):**
1. Push to `master` — GitHub Actions runs `pytest api/` and, on success, triggers a Render deploy hook
2. Monitor deploy at `https://dashboard.render.com` → service `rat-api`
3. Verify with `curl /health`

**Frontend (Vercel):**
1. Push to `master` — Vercel auto-deploys on every push
2. Preview URL is posted to the GitHub commit status
3. Production at `https://web-beige-three-56.vercel.app`

**Rollback:**
- Render: go to the service dashboard → "Events" → click any previous deploy → "Redeploy"
- Vercel: `vercel rollback` or promote a previous deployment from the dashboard

---

## Degraded mode (API returns 503)

The API starts in degraded mode when the CatBoost model artifact cannot be loaded. This produces:
```
INFO: starting in degraded mode
model_loaded: false
```

**Root cause A — artifact missing from the container:**
```bash
# On Render shell or locally:
ls ml/artifacts/registry.json
ls ml/artifacts/tabular/catboost/
```
The 1.2 MB CatBoost model at `ml/artifacts/tabular/catboost/2026-06-05T18-42-14/` must be committed to git (force-added despite `.gitignore`). If it's missing, re-add with:
```bash
git add -f ml/artifacts/registry.json ml/artifacts/tabular/catboost/2026-06-05T18-42-14/
git commit -m "chore: re-add CatBoost artifact"
git push
```

**Root cause B — path resolution failure:**
`loader.py` uses `Path(__file__).resolve().parents[4]` as repo root. If the package layout changes, update the `parents[N]` index. Verify with:
```python
from rat_api.ml.loader import _REPO_ROOT
print(_REPO_ROOT)  # should be repo root
```

---

## Cold start / keepalive

Render free tier spins down after 15 min of inactivity. First request after sleep takes ~30s.

A keepalive GitHub Action (`.github/workflows/keepalive.yml`) pings `/health` every 14 minutes using `workflow_dispatch` on a cron schedule. If the keepalive stops working, the API will cold-start on the next user request — not a hard failure, just slow.

To check whether the keepalive is running: GitHub → Actions → "Keepalive" → most recent run timestamp.

---

## Database

**Connection strings (in env):**
- `DATABASE_URL` — pooled connection via PgBouncer (for most queries)
- `DIRECT_DATABASE_URL` — direct connection (for migrations, `COPY`, long-running ingest)

**Useful queries:**

```sql
-- Check prediction table coverage
SELECT COUNT(*), MIN(week_start), MAX(week_start)
FROM app.risk_predictions;
-- expect: 19157 rows, 2023-05-22 to 2026-05-11

-- Check RAG chunk count
SELECT COUNT(*) FROM app.rag_chunks;
-- expect: ~1190

-- Check NTA boundaries loaded
SELECT COUNT(*) FROM raw.nta_boundaries;
-- expect: 223
```

**Migrations:**
```bash
uv run --package rat-api alembic upgrade head
```

---

## Re-materializing predictions

If new training data is added or the model is retrained, re-run:
```bash
uv run python ml/scripts/materialize_predictions.py
```
This upserts all 223 NTAs × 156 weeks into `app.risk_predictions`. Takes ~5 minutes.

---

## Environment variables

| Variable | Where | Description |
|---|---|---|
| `DATABASE_URL` | Render + local `.env` | Pooled Supabase connection |
| `DIRECT_DATABASE_URL` | Render + local `.env` | Direct Supabase connection (ingest/migration) |
| `GROQ_API_KEY` | Render | LLM generation |
| `DISABLE_RERANKER` | Render (`=1`) | Disables BGE-M3 + Reranker (free tier) |
| `ML_ARTIFACTS_DIR` | Optional | Default: `ml/artifacts` (relative to repo root) |
| `SENTRY_DSN` | Render | Error tracking (optional; safe to omit) |
| `NEXT_PUBLIC_API_URL` | Vercel | Points frontend at API |
| `NEXT_PUBLIC_SENTRY_DSN` | Vercel | Frontend error tracking (optional) |

---

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `/health` returns 503 | Model in degraded mode | See "Degraded mode" section |
| Map shows no dots | API returning empty array for the selected week | Check week_start is ≤ 2026-05-11; check `app.risk_predictions` has rows |
| Chat returns no citations | BM25 retrieval returned 0 results | Query too short or too specific; RAG chunks may not cover the topic |
| Frontend shows "Could not load risk data" | API cold starting (Render spin-down) | Wait 30s and refresh; or check if Render deploy failed |
| GitHub Actions not triggering | Workflow watching wrong branch | All workflows should target `master`; verify in `.github/workflows/*.yml` |
