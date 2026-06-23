# NYC Rat Risk Intelligence

> Neighborhood-level rat risk prediction and cited legal Q&A over NYC public data.

**Live demo:** [web-beige-three-56.vercel.app](https://web-beige-three-56.vercel.app)  
**API:** [rat-api-g3lf.onrender.com](https://rat-api-g3lf.onrender.com/health)

---

## What this is

A deployed multi-modal risk prediction and retrieval system. Users see a risk choropleth of NYC at
the NTA (Neighborhood Tabulation Area) level, can drill into any neighborhood for factor attributions
and a 12-week forecast, and can ask natural-language questions answered from the NYC Health Code with
cited legal sources.

**Supervised label:** DOHMH rodent inspection `RESULT == 'Active Rat Signs'` — not 311 call volume.
This is an important distinction: the model predicts where inspectors will *find* active rat signs,
not where residents complain.

---

## Metrics

| Metric | Value |
|---|---|
| CatBoost test PR-AUC | **0.7947** |
| Full ensemble (fusion) PR-AUC | **0.7975** |
| TFT validation loss | **0.163** |
| Top-decile lift | **1.53×** |
| RAG corpus | 1,190 chunks (5 legal sources) |
| NTAs modelled | 223 |
| Panel weeks | 156 (May 2023 – May 2026) |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Next.js 16 (Vercel)               │
│  / map   /nta/[id] detail   /chat   /about          │
│  MapLibre GL JS + React Query + shadcn/ui            │
└────────────────────┬────────────────────────────────┘
                     │ HTTPS
┌────────────────────▼────────────────────────────────┐
│               FastAPI (Render free tier)             │
│  /risk/map   /risk/nta/{id}   /chat (SSE)           │
│  CatBoost + TFT + Fusion meta-learner               │
│  BGE-M3 + BM25 + RRF hybrid retrieval              │
└────────────────────┬────────────────────────────────┘
                     │ asyncpg
┌────────────────────▼────────────────────────────────┐
│         Supabase Free (Postgres 15)                  │
│  PostGIS · pgvector · pg_trgm                       │
│  raw / features / app schemas                       │
└─────────────────────────────────────────────────────┘
```

See [`docs/architecture.md`](docs/architecture.md) for the full diagram and [`SPEC.md`](SPEC.md) for the build specification.

---

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Python 3.12 + Pydantic v2 + asyncpg |
| Frontend | Next.js 16 + TypeScript + Tailwind 4 + MapLibre GL JS |
| Database | Supabase Free (Postgres 15 + PostGIS + pgvector) |
| Tabular model | CatBoost (primary) + LightGBM + LR baselines |
| Temporal model | Darts TFT + Chronos-2 challenger |
| Fusion | Stacked logistic regression meta-learner + isotonic calibration |
| RAG retrieval | BM25 (`tsvector`) + pgvector HNSW + RRF (k=60) |
| LLM | Groq (free tier, generation); BGE-M3 (embeddings, self-hosted) |
| Observability | OpenInference spans → JSONL sink |
| Hosting | Render free (API) + Vercel Hobby (web) + Supabase Free (DB) |

---

## Data sources

| Source | Rows | Notes |
|---|---|---|
| DOHMH Rodent Inspections | ~250k | 3-year window; primary label source |
| NYC 311 Rodent Complaints | 229k NTA-weeks | Aggregated at ingest |
| DOHMH Restaurant Inspections | 296k | Pest violation codes 04K/04L/08A |
| DOB Permits | 942k | Construction disturbance signal |
| PLUTO (DCP) | 858k lots | Static building features |
| NOAA/Meteostat Weather | 2,269 days | Central Park station |
| NYC Health Code (RAG corpus) | 1,190 chunks | 5 PDF sources |

---

## Quickstart

```bash
# Python environment
uv sync --all-packages

# Run API tests
uv run pytest api/ -m "not integration" -v

# Start API locally
export $(grep -v '^#' .env | xargs)
uv run --package rat-api uvicorn rat_api.main:app --reload --port 8000

# Start frontend locally
cd web && pnpm dev
```

---

## Known limitations

- **Free-tier cold starts**: Render spins down after 15 min of inactivity; first request after sleep takes ~30s.
- **Panel lag**: feature data ends May 2026; predictions beyond that date fall back to the most recent available week.
- **Clay embeddings not active**: Sentinel-2 rasters were ingested but the Clay v1.5 embedding pipeline was not run due to memory constraints; `clay_pca_*` columns are NULL and excluded from training.
- **BM25-only RAG in prod**: BGE-M3 vector search and BGE Reranker are disabled on Render free tier (512 MB RAM limit); chat uses BM25 retrieval only.
- **No custom domain**: using `*.vercel.app` and `*.onrender.com` subdomains.

---

## Repo structure

```
├── api/          FastAPI backend + Alembic migrations
├── web/          Next.js 16 frontend
├── ml/           Training scripts and model artifacts
├── evals/        RAG evaluation suite (50-item gold set)
├── docs/         Architecture, runbook, decisions (ADRs)
└── SPEC.md       Authoritative build specification
```

---

## License

MIT
