# NYC Rat Risk Intelligence Platform — Build Specification

> **How to use this document with Claude Code**
>
> This is the authoritative spec. When you disagree with it, flag the disagreement and propose an alternative — do not silently deviate. Work in the phase order defined in Section 14. Within each phase, generate a plan before writing code, wait for approval, then implement. When the spec underdetermines a choice, prefer the simpler and more conventional option, and document the decision in `docs/decisions/NNNN-title.md` as a short ADR. If a dependency version conflicts with what's specified here, flag it — do not silently bump.
>
> **Invariants that must never be violated**: (1) the supervised label is DOHMH rodent inspection `RESULT`, not 311 call volume; (2) all LLM/retrieval/rerank/generation calls are instrumented to OpenInference spec with dual OTLP export (Phoenix + JSONL file sink); (3) no LangChain or LangGraph imports (except `langchain-text-splitters` as a standalone utility); (4) all model training and evaluation uses time-series cross-validation, never random splits.

---

## 1. Project overview

**What this is.** A deployed, multi-modal risk prediction and retrieval system over NYC rat data. Users see a risk choropleth of NYC at the NTA (Neighborhood Tabulation Area) level, can drill into a specific NTA to see factor attributions, and can ask natural-language questions that are answered from the NYC Health Code with cited legal sources.

**Who it's for (conceptually).** Four hypothetical user personas that shape product decisions: DOHMH ops lead (wants to prioritize inspections), landlord (wants to understand compliance obligations), journalist (wants a story-ready data view), resident (wants to know their neighborhood).

**What "done" looks like at end of Week 6.** Public URL at a custom domain; Next.js frontend deployed to Vercel; FastAPI backend deployed to Fly.io; Supabase Pro Postgres with PostGIS and pgvector; trace export running to both a local Phoenix instance and a JSONL file sink; Sentry catching errors; Better Stack status page live; load test run at 50 concurrent users with p50/p95/p99 published; RAG eval suite of 50 gold Q&A items with Recall@5, faithfulness, and citation accuracy published in the README; Loom demo linked; `docs/runbook.md`, `docs/cost-model.md`, `docs/PRD.md`, `docs/case-study.md` all written; Terraform config at `infra/terraform/aws/` deploying the same Docker image to AWS ECS; case-study blog post drafted.

**What this project is not.** It is not a research project. It is not an attempt to beat a state-of-the-art benchmark. It is not a notebook. It is not a chat-with-my-PDFs clone.

---

## 2. Non-negotiable architectural decisions

These are locked. Do not propose alternatives during implementation unless you hit a concrete blocker.

| Layer | Decision |
|---|---|
| **Supervised label** | DOHMH rodent inspection `RESULT == 'Active Rat Signs'` at BBL-day grain, aggregated to NTA-week |
| **Tabular model** | CatBoost (primary), LightGBM and logistic regression as ablation baselines, TabPFN v2 for small-borough subsets |
| **Temporal model** | Temporal Fusion Transformer via Darts; Chronos-2 fine-tune as challenger ensemble member |
| **Vision track** | Sentinel-2 L2A via Microsoft Planetary Computer → frozen Clay v1.5 → 32-dim PCA → concat into CatBoost features |
| **Fusion** | Late-fusion stacked meta-learner (logistic regression) over per-modality out-of-fold predictions, with isotonic calibration |
| **RAG corpus** | NYC Health Code Title 24 Article 151 + HMC §§27-2017 through 27-2018.1 + 24 RCNY §81.23 + ECB/OATH penalty schedule + DOHMH Rodent Academy PDFs |
| **Chunking** | Section-aware hierarchical (parse `§`, subsection markers), 400-600 token child chunks with 10-15% overlap, parents at 1500-2500 tokens, contextual prefixes |
| **Embeddings** | voyage-3-large primary; BGE-M3 self-host ablation |
| **Retrieval** | Hybrid BM25 (Postgres `tsvector` + `pg_trgm`) + dense (pgvector HNSW) + RRF fusion (k=60) + BGE Reranker v2-M3 self-hosted in the API container; Cohere Rerank 3.5 is an optional ablation that runs only if `COHERE_API_KEY` is present in env |
| **Vector store** | pgvector inside Supabase Postgres (same instance as PostGIS) |
| **LLM generation** | Claude Haiku 4.5 for `/chat` generation (default); Claude Sonnet 4.5 used only for the 50-item gold eval judge. Every LLM call logs token count and USD cost to prevent credit overrun. |
| **LLM orchestration** | Raw SDK calls (`anthropic`, `voyageai`, `cohere`). `litellm` as thin router. No LangChain. (`openai` only for eval judge; swap for Claude Opus if cost becomes an issue.) |
| **Backend** | FastAPI + Pydantic v2 + uvicorn, Python 3.12 |
| **Frontend** | Next.js 16 App Router + TypeScript + Tailwind + shadcn/ui + MapLibre GL JS + deck.gl |
| **Database** | Supabase Free: Postgres 15 + PostGIS + pgvector (500 MB limit; managed by data-reduction constraints in Section 5.1) |
| **Primary deployment** | Vercel (frontend, free Hobby tier, `*.vercel.app` subdomain) + Render free tier (backend, `*.onrender.com` subdomain) + Supabase Free (database) |
| **Secondary deployment** | None — see `docs/decisions/0001-single-cloud.md`; Docker image is portable if future deployment is needed |
| **Observability (LLM)** | OpenInference-spec instrumentation, dual OTLP export to Arize Phoenix (self-hosted Docker) + JSONL file sink |
| **Observability (app)** | Sentry free tier (errors) + Better Stack free tier (uptime, status page) |
| **CI/CD** | GitHub Actions: per-service test/deploy workflows + nightly RAG eval workflow + 6-day `/health` ping cron to prevent Supabase/Render idle pauses |
| **IaC** | Terraform for Sentry project only (no custom domain; no AWS sidecar) |
| **Package managers** | `uv` for Python, `pnpm` for JS, in a monorepo |

---

## 3. Repository structure

```
nyc-rat-risk/
├── README.md                    # Public-facing: demo link, metrics, architecture diagram
├── SPEC.md                      # This file; evolves as decisions are made
├── pyproject.toml               # Workspace-level uv config
├── pnpm-workspace.yaml          # Workspace-level pnpm config
├── docker-compose.yml           # Full local stack: api, phoenix, postgres-with-extensions
├── .env.example                 # All required env vars with placeholder values
├── .github/
│   └── workflows/
│       ├── api-test-deploy.yml
│       ├── web-test-deploy.yml
│       └── rag-eval-nightly.yml
├── api/                         # FastAPI backend
│   ├── pyproject.toml
│   ├── Dockerfile
│   ├── render.yaml
│   ├── src/
│   │   ├── rat_api/
│   │   │   ├── __init__.py
│   │   │   ├── main.py          # FastAPI app + lifespan
│   │   │   ├── config.py        # Pydantic Settings
│   │   │   ├── db.py            # Async SQLAlchemy + asyncpg
│   │   │   ├── models/          # Pydantic request/response models
│   │   │   ├── routes/
│   │   │   │   ├── risk.py      # /risk/* endpoints
│   │   │   │   ├── chat.py      # /chat endpoint (RAG)
│   │   │   │   ├── inspections.py
│   │   │   │   └── health.py
│   │   │   ├── ml/
│   │   │   │   ├── loader.py    # Lifespan-managed model loading
│   │   │   │   ├── predict.py   # Inference pipeline
│   │   │   │   └── features.py  # Online feature assembly
│   │   │   ├── rag/
│   │   │   │   ├── retriever.py # Hybrid BM25+dense+RRF+rerank
│   │   │   │   ├── generator.py # Claude prompt + streaming
│   │   │   │   └── prompts.py
│   │   │   ├── obs/
│   │   │   │   ├── tracing.py   # OpenInference + dual OTLP exporters
│   │   │   │   └── spans.py     # Span helpers
│   │   │   └── utils/
│   │   └── tests/
│   └── scripts/                 # One-off admin scripts
├── web/                         # Next.js frontend
│   ├── package.json
│   ├── next.config.ts
│   ├── tsconfig.json
│   ├── tailwind.config.ts
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx             # Landing + map
│   │   ├── nta/[id]/page.tsx    # NTA detail
│   │   ├── chat/page.tsx        # RAG chat
│   │   └── api/                 # BFF if needed; prefer direct backend calls
│   ├── components/
│   │   ├── map/
│   │   ├── charts/
│   │   ├── chat/
│   │   └── ui/                  # shadcn components
│   └── lib/
│       ├── api.ts               # Typed backend client
│       └── types.ts
├── ml/                          # ML training and evaluation (not served)
│   ├── pyproject.toml
│   ├── notebooks/               # EDA only; no production code here
│   ├── src/
│   │   ├── data/
│   │   │   ├── ingest_311.py
│   │   │   ├── ingest_rodent_inspections.py
│   │   │   ├── ingest_restaurant_inspections.py
│   │   │   ├── ingest_dob_permits.py
│   │   │   ├── ingest_pluto.py
│   │   │   ├── ingest_weather.py
│   │   │   ├── ingest_sentinel2.py
│   │   │   ├── bbl_join.py      # The one that matters most
│   │   │   └── tract_crosswalk.py
│   │   ├── features/
│   │   │   ├── spatial_lags.py
│   │   │   ├── temporal.py
│   │   │   ├── regime_indicators.py
│   │   │   └── clay_embeddings.py
│   │   ├── models/
│   │   │   ├── tabular.py       # CatBoost, LGB, LR baselines, TabPFN
│   │   │   ├── temporal.py      # TFT, Chronos-2
│   │   │   ├── fusion.py        # Stacked meta-learner + isotonic
│   │   │   └── registry.py      # Versioned model artifacts
│   │   ├── eval/
│   │   │   ├── timeseries_cv.py
│   │   │   ├── metrics.py       # PR-AUC, top-decile lift, calibration
│   │   │   └── reports.py       # Ablation tables → markdown
│   │   └── training/
│   │       ├── train_tabular.py
│   │       ├── train_tft.py
│   │       └── train_fusion.py
│   └── tests/
├── evals/                       # RAG evals (Evalforge will plug in here)
│   ├── pyproject.toml
│   ├── gold/
│   │   └── article151_qa_v1.jsonl   # 50-item gold set
│   ├── src/
│   │   ├── judges/              # LLM-as-judge prompts and runners
│   │   ├── metrics/             # Recall@K, faithfulness, citation accuracy
│   │   ├── runners.py           # Orchestrates eval runs against /chat
│   │   └── report.py            # Markdown output for README badge
│   └── results/                 # Gitignored artifacts; CI uploads to issues
├── infra/
│   ├── terraform/
│   │   ├── primary/             # Cloudflare, Supabase, Sentry
│   │   └── aws/                 # ECS sidecar deployment
│   └── scripts/
├── docs/
│   ├── architecture.md
│   ├── decisions/               # ADRs
│   ├── runbook.md
│   ├── cost-model.md
│   ├── PRD.md
│   ├── case-study.md
│   └── diagrams/                # source + rendered
└── data/                        # Gitignored; local cache of ingested sources
```

---

## 4. Environment and configuration

### 4.1 Required environment variables

All required env vars, with `.env.example` committed and real `.env` gitignored:

```
# Socrata (NYC Open Data)
NYC_SOCRATA_APP_TOKEN=

# Weather
NOAA_API_TOKEN=                    # Optional; meteostat works without

# Embeddings + rerank + LLMs
VOYAGE_API_KEY=
COHERE_API_KEY=
ANTHROPIC_API_KEY=
OPENAI_API_KEY=                    # For eval judges only; swap if cost becomes an issue

# Database
DATABASE_URL=                       # Supabase pooled connection string
DIRECT_DATABASE_URL=                # Supabase direct connection (for migrations)

# Observability
OTEL_EXPORTER_OTLP_ENDPOINT=        # Phoenix OTLP in dev; prod disabled or separate
OTEL_SERVICE_NAME=rat-api
OBS_JSONL_PATH=/var/log/rat-api/traces.jsonl
SENTRY_DSN=

# Geospatial data
PLANETARY_COMPUTER_KEY=             # Optional; most endpoints anonymous

# Frontend (NEXT_PUBLIC_*)
NEXT_PUBLIC_API_BASE_URL=
NEXT_PUBLIC_MAPTILER_KEY=
NEXT_PUBLIC_SENTRY_DSN=
```

### 4.2 Python environment

Use `uv` everywhere. One workspace `pyproject.toml` at repo root declares `api`, `ml`, and `evals` as workspace members so they share a lockfile.

```bash
uv sync --all-packages            # Install everything
uv run pytest                      # Run tests
uv run --package rat-api uvicorn rat_api.main:app --reload
```

Python pinned at 3.12. Key top-level deps:
- `fastapi`, `uvicorn[standard]`, `pydantic>=2.6`, `pydantic-settings`
- `sqlalchemy[asyncio]>=2.0`, `asyncpg`, `alembic`
- `anthropic`, `openai`, `voyageai`, `cohere`, `litellm`
- `opentelemetry-sdk`, `opentelemetry-exporter-otlp`, `openinference-instrumentation-*` (anthropic, openai, cohere)
- `arize-phoenix` (for self-hosted dev UI; runs as a separate container, not imported)
- `sentry-sdk[fastapi]`
- `catboost`, `lightgbm`, `scikit-learn`, `tabpfn`
- `darts[torch]`, `pytorch-lightning`, `transformers`
- `sodapy`, `meteostat`, `rasterio`, `rioxarray`, `pystac-client`, `planetary-computer`, `stackstac`
- `libpysal`, `geopandas`, `shapely`, `pyproj`
- `langchain-text-splitters` (standalone utility only — no other LangChain packages)

### 4.3 JS environment

pnpm workspace. Node 22 LTS. Key deps:
- `next@16`, `react@19`, `typescript@5.6+`
- `tailwindcss@4`, `shadcn-ui` components installed locally
- `maplibre-gl`, `react-map-gl`, `deck.gl`, `@deck.gl/mapbox`, `@deck.gl/geo-layers`
- `@tanstack/react-query`, `zod`
- `recharts` for attribution and time-series charts
- `@sentry/nextjs`

---

## 5. Data layer

### 5.1 Datasets (primary sources of truth)

All ingest scripts are idempotent, resumable, and write to the `raw` schema in Postgres.

| Script | Source | Socrata ID | Grain | Refresh |
|---|---|---|---|---|
| `ingest_rodent_inspections.py` | NYC DOHMH Rodent Inspection | `p937-wjvj` (verify, try `jh4g-rp64` as fallback) | BBL-day; **cap to last 3 years on ingest** to stay within Supabase Free 500 MB limit | Full refresh weekly |
| `ingest_311.py` | NYC 311 Service Requests, filter `complaint_type='Rodent'` | `erm2-nwe9` | **Aggregate to NTA-week counts during ingest** — do not store per-complaint rows. Write directly to `raw.complaints_nta_week (nta_id, week_start, complaint_count)`. | Daily incremental via `updated_at` cursor |
| `ingest_restaurant_inspections.py` | DOHMH Restaurant Inspections | `43nn-pn8j` | Per-inspection, codes 04K/04L/08A flagged | Weekly |
| `ingest_dob_permits.py` | DOB NOW + BIS permits | `ipu4-2q9a` + `rbx6-tga4` | Per-permit | Daily incremental |
| `ingest_pluto.py` | DCP MapPLUTO | DCP direct download, latest `25v4+` | Per-BBL (~860k) | Quarterly snapshot |
| `ingest_weather.py` | NOAA GHCN-Daily (`USW00094728` Central Park) | via `meteostat` | Daily | Daily |
| `ingest_sentinel2.py` | Sentinel-2 L2A via MPC STAC | tiles 18TWL, 18TWK | Per-NTA quarterly mosaic | Quarterly |

Socrata rate limit: ~1000 req/hr authenticated, page size up to 50,000. Use `sodapy.Socrata` with `$limit=50000` and `$offset` pagination. **All ingest scripts emit OpenTelemetry spans so ingestion runs appear in Phoenix alongside inference traces.**

### 5.2 Postgres schema

Three schemas in one Supabase database:

- `raw` — ingested data as-is (with added `ingested_at` timestamps)
- `features` — joined, typed, feature-engineered tables at BBL-week and NTA-week grain
- `app` — application-facing tables (risk predictions cache, chat sessions, eval runs)

Extensions enabled in the initial migration:
```sql
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
```

Core tables (full DDL to be generated in Phase 1; schemas below are illustrative):

```sql
-- raw.rodent_inspections
CREATE TABLE raw.rodent_inspections (
  inspection_id       TEXT PRIMARY KEY,
  inspection_date     DATE NOT NULL,
  bbl                 TEXT,                    -- 10-char, zero-padded
  bin                 TEXT,
  borough             SMALLINT,
  block               INTEGER,
  lot                 INTEGER,
  result              TEXT NOT NULL,            -- 'Active Rat Signs' | 'Passed' | ...
  inspection_type     TEXT,
  job_progress        SMALLINT,
  geom                GEOMETRY(Point, 4326),
  ingested_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX ON raw.rodent_inspections USING GIST (geom);
CREATE INDEX ON raw.rodent_inspections (bbl);
CREATE INDEX ON raw.rodent_inspections (inspection_date);

-- features.nta_week_panel
CREATE TABLE features.nta_week_panel (
  nta_id              TEXT NOT NULL,            -- NTA 2020 code
  week_start          DATE NOT NULL,            -- Monday
  -- Label
  active_rat_signs_count INTEGER NOT NULL DEFAULT 0,
  inspections_count   INTEGER NOT NULL DEFAULT 0,
  active_rat_signs_rate NUMERIC,                -- = count / inspections
  active_rat_signs_ind  BOOLEAN,                -- = count > 0
  -- 311 features (lagged)
  complaints_count    INTEGER NOT NULL DEFAULT 0,
  complaints_lag_1w   INTEGER,
  complaints_lag_4w   INTEGER,
  complaints_lag_12w  INTEGER,
  -- Restaurant
  rest_pest_violations_count INTEGER NOT NULL DEFAULT 0,
  -- Permits (disturbance)
  permits_active_count INTEGER NOT NULL DEFAULT 0,
  demolitions_count   INTEGER NOT NULL DEFAULT 0,
  -- Weather
  weather_tavg_c      NUMERIC,
  weather_prcp_mm     NUMERIC,
  weather_hdd         NUMERIC,
  weather_cdd         NUMERIC,
  -- Static (PLUTO aggregates)
  units_total         INTEGER,
  year_built_median   INTEGER,
  landuse_residential_pct NUMERIC,
  landuse_commercial_pct  NUMERIC,
  -- Spatial lags (populated by features/spatial_lags.py)
  neighbor_active_rat_signs_rate_lag_1w NUMERIC,
  neighbor_complaints_count_lag_4w      NUMERIC,
  -- Regime indicators
  regime_covid        BOOLEAN,                  -- 2020-03 to 2020-06
  regime_8pm_setout   BOOLEAN,                  -- from 2023-04
  regime_commercial_containerization BOOLEAN,   -- from 2023-07
  regime_residential_containerization BOOLEAN,  -- from 2024-11
  regime_rmz_active   BOOLEAN,
  -- Clay embedding (32-dim PCA)
  clay_pca_0          NUMERIC, ..., clay_pca_31 NUMERIC,
  PRIMARY KEY (nta_id, week_start)
);

-- app.health_code_chunks
CREATE TABLE app.health_code_chunks (
  chunk_id            UUID PRIMARY KEY,
  document            TEXT NOT NULL,            -- 'hc_article_151', 'hmc_27_2017', etc.
  citation            TEXT NOT NULL,            -- '§151.02(a)(2)'
  authority           TEXT NOT NULL,            -- 'NYC DOHMH', 'NYC HPD', ...
  section_path        TEXT[] NOT NULL,          -- ['151', '151.02', '151.02(a)', '151.02(a)(2)']
  defined_terms       TEXT[],                   -- extracted via regex
  cross_refs          TEXT[],                   -- extracted citation references
  parent_chunk_id     UUID REFERENCES app.health_code_chunks,
  content             TEXT NOT NULL,
  content_with_prefix TEXT NOT NULL,            -- 'From NYC Health Code §151...: <content>'
  token_count         INTEGER NOT NULL,
  effective_date      DATE,
  version_hash        TEXT NOT NULL,
  embedding           VECTOR(1024),             -- voyage-3-large dim
  embedding_bge       VECTOR(1024),             -- BGE-M3 for ablation
  content_tsv         TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);
CREATE INDEX ON app.health_code_chunks USING HNSW (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
CREATE INDEX ON app.health_code_chunks USING GIN (content_tsv);
CREATE INDEX ON app.health_code_chunks USING GIN (content gin_trgm_ops);

-- app.chat_sessions
CREATE TABLE app.chat_sessions (
  session_id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  user_fingerprint    TEXT
);

-- app.chat_messages
CREATE TABLE app.chat_messages (
  message_id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_id          UUID NOT NULL REFERENCES app.chat_sessions,
  role                TEXT NOT NULL,            -- 'user' | 'assistant' | 'system'
  content             TEXT NOT NULL,
  retrieved_chunks    JSONB,                    -- array of chunk_ids + scores
  trace_id            TEXT,                     -- OpenInference trace
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  latency_ms          INTEGER,
  cost_usd            NUMERIC(10, 6)
);

-- app.risk_predictions
CREATE TABLE app.risk_predictions (
  nta_id              TEXT NOT NULL,
  predicted_for_week  DATE NOT NULL,
  risk_score          NUMERIC NOT NULL,         -- calibrated probability
  risk_decile         SMALLINT NOT NULL,
  top_factors         JSONB NOT NULL,           -- [{feature, contribution}]
  model_version       TEXT NOT NULL,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (nta_id, predicted_for_week, model_version)
);
```

### 5.3 Feature engineering requirements

- **BBL join**: always cast BBL to 10-char zero-padded string. Handle condo billing-BBL via `PLUTO.APPBBL` when `BBL != APPBBL`. Emit a data-quality report showing unmatched BBL count per source.
- **Tract crosswalk**: for any historical feature crossing 2019→2020, apply 2010→2020 NTA crosswalk with area-weighted allocation.
- **Projections**: raw ingestion preserves NY State Plane Long Island Feet (EPSG:2263) when present; features schema uses WGS84 (EPSG:4326); `pyproj.Transformer` for conversions. **Write a test that fails if any `geom` column contains coordinates outside the NYC bounding box** to catch projection bugs early.
- **Time-series splits**: partition training using expanding-window CV with minimum-28-day gap between train end and validation start to prevent leakage through reporting lag. Final holdout is the most recent 12 weeks.
- **Regime indicators**: hardcoded boolean feature columns for COVID period, 8PM set-out (2023-04-01 onward), commercial containerization (2023-07-31 onward, tightening 2024-03-01), residential containerization (2024-11 onward by zone), and RMZ active status. These are inputs to the model, not a preprocessing filter.

---

## 6. ML layer

### 6.1 Training pipeline

Every training script writes to `ml/artifacts/<model_name>/<timestamp>/` and updates a `registry.json` with the latest version. The FastAPI service loads models by name from the registry via an env var pointing to a directory or an S3 bucket.

**Tabular model (`train_tabular.py`)**:
- Target: `active_rat_signs_ind` at NTA-week grain
- Features: all non-embedding columns in `features.nta_week_panel` + Clay PCA columns when available + neighbor spatial lags
- Model: CatBoost (primary), LightGBM (ablation), logistic regression with L2 (baseline)
- Training: expanding-window time-series CV, 5 folds, with 28-day gap
- Output: calibrated probability (isotonic calibration on the final fold)
- Per-borough TabPFN v2 models for boroughs with ≤10k training rows
- Artifacts: model binary, feature importance, SHAP values for top-20 features per prediction, fold-by-fold PR-AUC, final test-period PR-AUC, top-decile lift

**Temporal model (`train_tft.py`)**:
- Target: `active_rat_signs_count` at NTA-week grain, 12-week forecast horizon
- Static covariates: borough, NTA area, PLUTO aggregates
- Past covariates: weather actuals, 311 lags, prior inspection outcomes, regime indicators
- Future known covariates: weather forecast (use historical actuals as a proxy for training), calendar features, known scheduled demolitions
- Model: Darts `TFTModel`; Chronos-2 fine-tune as challenger
- Output: point forecast per NTA, percentile intervals
- **Document explicitly** in the training log why TFT over LSTM and why Chronos-2 for fine-tune

**Fusion model (`train_fusion.py`)**:
- Out-of-fold predictions from tabular and temporal models, plus Clay PCA features
- Meta-learner: logistic regression (primary), shallow MLP (ablation)
- Isotonic calibration on the meta-learner output
- Final artifact is a single callable that takes an NTA-week feature row and returns calibrated probability + top-N SHAP factors

### 6.2 Serving pipeline

The FastAPI backend loads all three models in `lifespan` (startup) and keeps them warm. **No re-training happens in the API container.** Endpoints:

```
GET /risk/nta/{nta_id}
  → {
      nta_id: str,
      current_week: date,
      risk_score: float,             # calibrated probability 0-1
      risk_decile: int,              # 1-10
      top_factors: [
        {feature: str, contribution: float, direction: "up"|"down", readable: str}
      ],
      model_version: str,
      forecast_12w: [{week: date, risk_score: float, ci_low: float, ci_high: float}]
    }

GET /risk/map?week={week}
  → [
      {nta_id: str, risk_score: float, risk_decile: int}
    ]    # ~200 NTAs, cached at CDN edge

GET /inspections/nta/{nta_id}?since={date}
  → [
      {inspection_id: str, date: date, result: str, bbl: str, lat: float, lon: float}
    ]

POST /chat
  body: {session_id: uuid?, message: str}
  → SSE stream of {type: "chunk"|"citation"|"done", data: ...}

GET /health
  → {status: "ok", model_version: str, db_latency_ms: int, git_sha: str}
```

**Feature assembly at serve time** uses materialized features joined to the current week's panel row. If an NTA has no current-week row (a gap in ingestion), return a 503 with a clear message; do not silently fabricate features.

---

## 7. RAG layer

### 7.1 Corpus ingestion

Write one script per source PDF:
- `ingest_health_code_151.py` — NYC Health Code Title 24 Article 151
- `ingest_hmc_27_2017.py` — NYC Housing Maintenance Code §§27-2017 through 27-2018.1
- `ingest_rcny_81_23.py` — 24 RCNY §81.23 (IPM for food establishments)
- `ingest_ecb_penalties.py` — ECB/OATH penalty schedule
- `ingest_rodent_academy.py` — DOHMH Rodent Academy PDFs

Each script: PDF → structured text (use `pypdf` or `pdfplumber`; fall back to OCR only if a PDF is image-based) → parse legal hierarchy via regex (`§\d+\.\d+`, `(a)`, `(1)`, `(i)`, etc.) → produce chunk records.

**Chunking algorithm**:
1. Parse by legal structure first. Each leaf subsection becomes a child chunk if ≤600 tokens; split longer subsections into overlapping 400-600 token windows.
2. Each section (e.g. `§151.02`) becomes a parent chunk.
3. For each chunk, prepend a contextual prefix: `"From <authority> <document> <citation>: <content>"`.
4. Extract defined terms from chunks marked as "Definitions" sections into a terms dictionary.
5. Extract cross-references (regex for citation patterns) and store in `cross_refs` array.
6. Compute `version_hash` as SHA-256 of `content_with_prefix`.

**Embedding**:
- Primary: Voyage `voyage-3-large`, `input_type="document"` at ingest.
- Ablation: BGE-M3 via `sentence-transformers`, stored in a second column.
- Batch size: 128 chunks per API call, with exponential backoff on 429s.

### 7.2 Retrieval pipeline

Implemented in `api/src/rat_api/rag/retriever.py`. One public function:

```python
async def retrieve(
    query: str,
    *,
    top_k_dense: int = 30,
    top_k_bm25: int = 30,
    top_k_after_rrf: int = 40,
    top_k_final: int = 6,
    expand_parents: bool = True,
) -> list[RetrievedChunk]
```

Steps:
1. **Query rewriting** (optional LLM step; log the rewrite in span attributes). Use Claude Haiku with a prompt that expands colloquial terms to statutory vocabulary. Cap at 200 tokens.
2. **Dense retrieval**: embed rewritten query with Voyage `input_type="query"`, pgvector HNSW cosine search.
3. **BM25 retrieval**: Postgres `plainto_tsquery` ranked with `ts_rank_cd`. Consider `paradedb` if time permits.
4. **RRF fusion**: reciprocal rank fusion with `k=60`.
5. **Rerank**: BGE Reranker v2-M3 (self-hosted in the API container, loaded at lifespan startup) on top-40 unique chunks, take top-6. If `COHERE_API_KEY` is present in env, Cohere Rerank 3.5 is used instead (ablation path).
6. **Parent expansion**: for each final chunk, include its parent chunk in the context if `expand_parents=True` and parent is within the context budget.

All steps emit OpenInference spans (see Section 9).

### 7.3 Generation

Implemented in `api/src/rat_api/rag/generator.py`. Uses **Claude Haiku 4.5** (default, ~10× cheaper than Sonnet) with the following system prompt contract (stored in `rag/prompts.py`). The eval judge in `evals/` uses Claude Sonnet 4.5 for higher-quality faithfulness scoring.

```
You are a legal assistant answering questions about NYC rodent regulations. You
MUST cite every factual claim using the `§<citation>` format provided in the
retrieved chunks. If the answer is not supported by the retrieved chunks, say
so explicitly; do not speculate. Format: one short answer paragraph, followed
by a "Sources:" list of citations with brief quotes.
```

Stream the response as SSE. Emit one span per LLM call. Record token counts and computed cost (use a hardcoded price table, keep it updated in `config.py`).

---

## 8. Frontend

### 8.1 Pages

- `/` — landing page. Hero with risk choropleth of NYC at the NTA level, time slider for the last 52 weeks, basic legend, and short explanation of the supervised label framing ("This model predicts where DOHMH inspectors are most likely to find active rat signs, based on ..."). A "See the case study" link.
- `/nta/[id]` — NTA detail. Larger map centered on the NTA, current risk score with decile badge, 12-week forecast chart with CI band, top-factors bar chart with readable labels, recent inspection outcomes list, nearby 311 complaints on the map.
- `/chat` — RAG chat. Thread UI with streamed responses, citations rendered as clickable pills that expand to show the quoted chunk, example prompts.
- `/about` — short explanation of methodology, data sources, limitations, and links to the blog post and GitHub.

### 8.2 Map stack

- MapLibre GL JS with MapTiler basemap (free 100K/month).
- `react-map-gl/maplibre` for React bindings.
- `deck.gl` via `MapboxOverlay` with `interleaved: true`.
- Layers:
  - `GeoJsonLayer` for NTA boundaries with risk-score color ramp; pre-rendered TopoJSON served from Vercel edge.
  - `ScatterplotLayer` for recent inspection outcomes with result-based coloring.
- Time slider drives the `data` prop; memoize to avoid re-creating layers on each render.
- **Do not ship full GeoJSON (>2MB) to the browser on every load**; pre-render TopoJSON with quantization, host on Vercel edge, gzip.

### 8.3 Data fetching

- `@tanstack/react-query` for all backend calls.
- Typed API client in `lib/api.ts` using `zod` schemas that mirror the Pydantic response models.
- SSE streaming for `/chat` via native `EventSource`, wrapped in a custom hook.

### 8.4 Design notes

- shadcn/ui components only; no component library sprawl.
- Tailwind for everything; no CSS modules.
- Dark mode optional; skip if time-constrained.
- Accessibility: full keyboard nav on chat and map controls; ARIA labels on the map.

---

## 9. Observability

### 9.1 Tracing

Every LLM-adjacent operation emits OpenInference-spec spans with correct `openinference.span.kind` attributes:
- `CHAIN` — root span per user request
- `LLM` — query rewriting, generation
- `RETRIEVER` — dense and BM25 retrieval (one span each) and the fusion (third span)
- `RERANKER` — Cohere rerank
- `TOOL` — any future tool use

Required attributes on retriever spans:
- `retrieval.documents.N.document.id`
- `retrieval.documents.N.document.content`
- `retrieval.documents.N.document.score`
- `retrieval.documents.N.document.metadata.citation`
- `retrieval.documents.N.document.metadata.authority`
- Custom: `retrieval.method` = `"dense"|"bm25"|"rrf"`, `retrieval.top_k`, `retrieval.score_distribution.{min,p50,max}`

Required attributes on LLM spans:
- `llm.model_name`, `llm.provider`
- `llm.input_messages`, `llm.output_messages`
- `llm.token_count.prompt`, `llm.token_count.completion`, `llm.token_count.total`
- `llm.usd_cost` (computed from token counts + price table)

### 9.2 Export

Configure **two OTLP exporters in parallel** in `obs/tracing.py`:
1. Phoenix OTLP HTTP endpoint in dev (configurable via `OTEL_EXPORTER_OTLP_ENDPOINT`), disabled in prod by default.
2. **JSONL file sink** at `OBS_JSONL_PATH` that writes one JSON object per span with all attributes flattened. This is the Evalforge handoff — Evalforge's Phase 7 will consume this file.

Both exporters run via a `BatchSpanProcessor`. **Do not couple application code to Phoenix or any vendor SDK.**

### 9.3 Application observability

- Sentry on both frontend (`@sentry/nextjs`) and backend (`sentry-sdk[fastapi]`). Capture all unhandled exceptions. Sample rate 1.0 at launch.
- Better Stack uptime monitor hitting `/health` every 3 minutes; public status page at `status.<domain>`.
- `/health` returns 200 only if DB ping succeeds and all three models are loaded.

---

## 10. Testing

### 10.1 Required test coverage

- **Data layer**: schema tests (geom bounding box, BBL format, required columns); crosswalk tests (2010↔2020 NTA mapping sanity); ingestion idempotency (re-running should not duplicate).
- **ML layer**: time-series CV leak test (assert no date overlap between train/val); calibration test (Brier score on holdout); feature importance smoke test.
- **RAG layer**: chunking invariants (token counts in range, citations match regex, cross-refs resolve); retrieval smoke test (known Q returns expected chunk in top-10); embedding input_type test (assert query embeddings use `input_type="query"`).
- **API layer**: route contract tests (Pydantic response models match), `/health` returns 200 when all loaded, SSE format for `/chat`.
- **Frontend**: component tests for the chat thread and map wrapper; Playwright smoke test hitting the deployed URL that (1) loads the map, (2) opens `/chat`, (3) streams a response, (4) asserts at least one citation rendered.

### 10.2 Eval suite (the important one)

`evals/gold/article151_qa_v1.jsonl` — 50 hand-authored Q&A items covering six failure modes:
- Cross-reference following (question requires following a `§` citation)
- Defined-term collisions (term defined differently in HMC vs. Health Code)
- Section-boundary fragmentation (answer spans two sections)
- Citation accuracy (question where a wrong citation would be plausible)
- Multi-hop penalty lookup (violation → penalty table → amount)
- Vocabulary gap (colloquial query that requires statutory term expansion)

Each item has:
```json
{
  "id": "article151-q-001",
  "question": "If a landlord doesn't exterminate rats after receiving a violation, how much is the fine?",
  "expected_citations": ["§27-2018", "§27-2115", "ECB AH4D"],
  "must_cite_at_least_one_of": [["§27-2018"], ["AH4D"]],
  "must_not_say": ["I don't know", "not specified"],
  "reference_answer": "...",
  "failure_mode": "multi_hop_penalty_lookup"
}
```

Metrics:
- **Retrieval Recall@5** and **Recall@10** measured against `expected_citations`
- **Citation accuracy** via regex match on asserted citations in the generated answer
- **Faithfulness** via LLM-as-judge (Claude Opus, binary 0/1) with a rubric that checks each claim against retrieved chunks
- **Refusal calibration**: what fraction of "I don't know" responses were correct (question genuinely unanswerable)

The nightly GitHub Action runs the eval suite and posts metrics as a README badge. Results written to `evals/results/<date>.json`.

---

## 11. Deployment

### 11.1 Primary (Render + Vercel + Supabase Free) — $0/month

- **Backend**: Render free tier (`render.yaml` in `api/`). Free instance spins down after 15 minutes of inactivity with a ~30-second cold start on the next request. Document this tradeoff in `docs/runbook.md`. The 6-day GitHub Actions cron (Section 12.4) keeps both Render and Supabase from hitting their idle-pause thresholds.
- **Frontend**: Vercel Hobby tier, production branch = `main`, preview on every PR. Public URL: `<project>.vercel.app` (no custom domain).
- **Database**: Supabase Free (500 MB storage limit). Pooled connection string used by the API; direct connection only for migrations. Data-reduction constraints (3-year inspection cap, NTA-week-aggregated 311) keep the DB under limit.
- **DNS/TLS**: No custom domain. Vercel manages TLS for the frontend subdomain; Render manages TLS for the backend subdomain.
- **Backend public URL pattern**: `https://<service>.onrender.com`
- **Frontend public URL pattern**: `https://<project>.vercel.app`

### 11.2 Secondary deployment

None. The Docker image is portable (no vendor lock-in in the application code), but running a second cloud deployment is out of scope. See `docs/decisions/0001-single-cloud.md` for the rationale.

### 11.3 Phoenix

`docker-compose.yml` runs `arizephoenix/phoenix` at `:6006` for local dev. Not deployed to prod. Traces from prod go only to the JSONL sink; Phoenix is for development visualization.

---

## 12. CI/CD

Three GitHub Actions workflows:

### 12.1 `api-test-deploy.yml`
- Triggers: PR to `main` (test only), push to `main` (test + deploy).
- Steps: checkout → uv sync → ruff + mypy → pytest → build Docker image → (on main) deploy to Render via Render Deploy Hook.

### 12.2 `web-test-deploy.yml`
- Triggers: PR to `main` (test + Vercel preview), push to `main` (Vercel production deploy).
- Steps: checkout → pnpm install → tsc --noEmit → eslint → vitest → playwright smoke test against preview URL.

### 12.3 `rag-eval-nightly.yml`
- Triggers: scheduled daily at 06:00 UTC.
- Steps: hit production `/chat` with all 50 gold questions → compute metrics → write `evals/results/<date>.json` → update README badge via commit to a `metrics` branch → open an issue if any metric regresses >5% from the previous week's median.

### 12.4 `keepalive.yml`
- Triggers: scheduled every 6 days at 12:00 UTC (just under Supabase Free's 7-day idle pause and Render's inactivity spin-down).
- Steps: HTTP GET to `$RENDER_BACKEND_URL/health` → assert 200 → log response. Prevents both Supabase from pausing the project and Render from accumulating cold-start lag.
- Required secrets: `RENDER_BACKEND_URL`.

---

## 13. Cost model

**Project budget: $0 recurring.** All infrastructure runs on free tiers. The only non-zero spend is the one-time $5 Anthropic API credit used during development and eval.

Document in `docs/cost-model.md` as a spreadsheet-style markdown table. Targets at launch:

| Line item | Monthly cost at launch | Notes |
|---|---|---|
| Supabase Free | $0 | 500 MB storage; data-reduction constraints keep us under |
| Render free tier | $0 | 512 MB RAM; 15-min spin-down; cold start ~30s |
| Vercel Hobby | $0 | 100 GB bandwidth/month |
| Custom domain / DNS | $0 | Using `*.vercel.app` + `*.onrender.com` subdomains |
| MapTiler free tier | $0 | 100K map loads/month |
| Voyage AI free tier | $0 | 200M tokens/month; corpus ingest ~1M tokens total |
| BGE Reranker v2-M3 | $0 | Self-hosted in the API container |
| Cohere Rerank 3.5 | $0 | Optional ablation only; not in the default path |
| Claude Haiku 4.5 generation | ~$0.50–$1 one-time | Covered by $5 Anthropic credit; ~500 dev/eval queries |
| Claude Sonnet 4.5 (eval judge) | ~$1–$2 one-time | 50-item gold set × multiple eval runs |
| Sentry free tier | $0 | 5K errors/month |
| Better Stack free tier | $0 | 10 monitors |
| **Total recurring** | **$0/month** | |
| **One-time dev spend** | **≤$5** | Anthropic API credit |

**Credit burn guardrails**: every LLM call in `generator.py` and `evals/` logs `llm.usd_cost` (computed from token counts × hardcoded price table in `config.py`). A middleware check warns (log + Sentry alert) if cumulative spend in the current calendar day exceeds $1. Hard-stop is not implemented to avoid breaking live demos, but the alert gives time to intervene.

Publish the 10× and 100× traffic projections alongside (these would require paid tiers). Per-query median cost target: ≤$0.001 for risk predictions, ≤$0.005 for RAG queries (Haiku pricing).

---

## 14. Build order and milestones

Each phase has a concrete deliverable. Do not start the next phase until the current one is merged and deployed (or, for early phases, reproducible locally).

### Phase 1 — Data foundation (Week 1)

**Goal**: a materialized feature table with properly joined keys and a reporting-bias analysis notebook.

**Deliverables**:
- [ ] Supabase project provisioned with PostGIS, pgvector, pg_trgm, uuid-ossp
- [ ] Alembic migrations for `raw`, `features`, `app` schemas with all Section 5.2 tables
- [ ] All seven ingestion scripts working and idempotent
- [ ] `bbl_join.py` produces a joined BBL-week table with <1% unmatched rate
- [ ] `tract_crosswalk.py` handles 2010↔2020 NTA mapping
- [ ] `features/spatial_lags.py` populates queen-contiguity neighbor features
- [ ] `features/regime_indicators.py` adds regime boolean columns
- [ ] Notebook `ml/notebooks/01_reporting_bias.ipynb` showing 311-complaint-density vs. inspection-outcome rate by NTA income decile, with a clear finding about bias
- [ ] Geom-bounding-box test passes
- [ ] Data-quality report generated (`ml/artifacts/data_quality/<date>.md`)

**Acceptance check**: `uv run python -m ml.scripts.build_panel` produces `features.nta_week_panel` with >100k rows and no NULLs in required columns.

### Phase 2 — Tabular baseline (Week 2)

**Goal**: a calibrated tabular risk model served from FastAPI with published metrics.

**Deliverables**:
- [ ] `train_tabular.py` trains CatBoost, LightGBM, and LR baselines with expanding-window TS-CV
- [ ] Per-borough TabPFN v2 for boroughs with ≤10k rows
- [ ] Isotonic calibration applied and Brier score reported
- [ ] Model artifacts written to `ml/artifacts/tabular/<version>/`
- [ ] FastAPI `/risk/nta/{nta_id}` and `/risk/map` endpoints working
- [ ] Feature importance and SHAP top-20 in the response
- [ ] Test-period PR-AUC published in `ml/artifacts/tabular/<version>/report.md`
- [ ] Top-decile lift metric reported
- [ ] Ablation table committed: CatBoost / LightGBM / LR / (CatBoost + TabPFN) rows
- [ ] FastAPI service Dockerized and running locally via `docker compose up`

**Acceptance check**: `curl localhost:8000/risk/nta/MN01` returns a valid response with `risk_score`, `risk_decile`, and `top_factors`; PR-AUC ≥ best single-feature baseline by measurable margin.

### Phase 3 — Multi-modal ensemble (Week 3)

**Goal**: TFT + Chronos-2 + Clay embeddings integrated via late fusion.

**Deliverables**:
- [ ] `ingest_sentinel2.py` pulls quarterly mosaics for NYC from MPC
- [ ] `features/clay_embeddings.py` runs frozen Clay v1.5, PCA to 32 dims, joins into `features.nta_week_panel`
- [ ] `train_tft.py` trains TFT on NTA-week panel
- [ ] Chronos-2 fine-tune as challenger ensemble member
- [ ] `train_fusion.py` builds stacked meta-learner with isotonic calibration over OOF predictions
- [ ] `/risk/nta/{nta_id}` now returns `forecast_12w` with CI bands
- [ ] Ablation table extended: CatBoost / +TFT / +Clay / +Chronos / full ensemble
- [ ] Architecture diagram rendered and committed to `docs/architecture.md`

**Acceptance check**: full ensemble PR-AUC ≥ tabular-only + 2 points; forecast CI widths reasonable (no degenerate intervals).

### Phase 4 — RAG + observability (Week 4)

**Goal**: cited-answer chat endpoint over the NYC Health Code, fully instrumented.

**Deliverables**:
- [ ] All five corpus ingestion scripts parse their PDFs and write to `app.health_code_chunks`
- [ ] Section-aware hierarchical chunking with contextual prefixes
- [ ] Defined-terms and cross-refs extracted
- [ ] voyage-3-large embeddings generated; BGE-M3 ablation column populated
- [ ] Hybrid retrieval (tsvector BM25 + dense + RRF + Cohere Rerank) working in `rag/retriever.py`
- [ ] Parent-chunk expansion implemented
- [ ] `/chat` endpoint streams SSE with Claude Sonnet 4.5
- [ ] OpenInference instrumentation with dual OTLP export (Phoenix + JSONL)
- [ ] Phoenix running in `docker-compose.yml`, traces visible at `localhost:6006`
- [ ] `evals/gold/article151_qa_v1.jsonl` authored with 50 items covering six failure modes
- [ ] `evals/src/runners.py` runs the suite against `/chat`; metrics written to `evals/results/<date>.json`
- [ ] Recall@5, faithfulness, citation accuracy published

**Acceptance check**: `curl -N localhost:8000/chat` with a known-answerable question streams a response with at least one correct citation; eval suite Recall@5 ≥ 0.70 on first run.

### Phase 5 — Frontend + deployment (Week 5)

**Goal**: public URL at a custom domain with the full UX working.

**Deliverables**:
- [ ] Next.js 16 App Router project scaffolded
- [ ] shadcn/ui components installed
- [ ] Landing page with MapLibre + deck.gl risk choropleth and time slider
- [ ] `/nta/[id]` detail page with forecast chart and factor bars
- [ ] `/chat` page with streaming, citations rendered as pills
- [ ] `/about` page
- [ ] Typed API client in `lib/api.ts`
- [ ] Vercel deployment live (`*.vercel.app`)
- [ ] Render deployment live (`*.onrender.com`)
- [ ] `keepalive.yml` GitHub Actions cron deployed and verified
- [ ] Sentry on both sides, Better Stack uptime, public status page
- [ ] `k6` load test at 50 concurrent users; p50/p95/p99 recorded in `docs/load-test.md`

**Acceptance check**: public URL loads in <2s TTI; status page shows "All Systems Operational"; Sentry dashboard accessible.

### Phase 6 — Hardening, AWS sidecar, writeups (Week 6)

**Goal**: polish, documentation, and the multi-cloud story.

**Deliverables**:
- [ ] `docs/decisions/0001-single-cloud.md` written (deliberate scoping decision, portability note)
- [ ] `docs/runbook.md` written
- [ ] `docs/cost-model.md` with 1×/10×/100× projections
- [ ] `docs/PRD.md` with four user personas and key product decisions
- [ ] `docs/case-study.md` blog post draft
- [ ] `docs/architecture.md` with full diagram
- [ ] README polished: demo link, 3–5 min Loom embed, architecture diagram, metrics table, known failure modes section
- [ ] Nightly eval CI workflow running
- [ ] All ADRs written in `docs/decisions/`
- [ ] All tests passing in CI
- [ ] Lighthouse score ≥90 on the landing page

**Acceptance check**: a third-party reviewer can understand the project end-to-end from the README in under 10 minutes, find the live demo, run a query, and see an eval dashboard badge showing current metrics.

---

## 15. Success metrics (publish in README)

| Metric | Target |
|---|---|
| Test-period PR-AUC | ≥ best single baseline + 2 points |
| Top-decile capture of `Active Rat Signs` | ≥ 40% |
| RAG Recall@5 | ≥ 0.75 |
| RAG faithfulness (LLM-as-judge) | ≥ 0.85 |
| RAG citation accuracy (regex) | ≥ 0.95 |
| p95 latency `/risk/nta` | < 800ms |
| p95 latency `/chat` (end-to-end) | < 3s |
| Median cost per RAG query | ≤ $0.001 |
| Monthly infra cost at launch traffic | ≤ $50 |
| Lighthouse score (landing) | ≥ 90 |
| Uptime (2 weeks post-launch) | ≥ 99% |

---

## 16. Cut line (if schedule compresses)

Drop in this order; document each cut as an ADR:

1. Mapillary street-view segmentation (never started — Week 6 stretch only)
2. Chronos-2 fine-tune (keep TFT alone)
3. ~~AWS ECS Terraform sidecar~~ — **already cut** (see `docs/decisions/0001-single-cloud.md`)
4. Clay v1.5 Sentinel-2 track (TFT + CatBoost still counts as multi-modal when combined with restaurant-inspection channel)
5. BGE-M3 embedding ablation (keep voyage-3-large only)
6. Cohere Rerank ablation (BGE Reranker v2-M3 self-hosted is already the default; skip the Cohere comparison entirely if time is short)

**Never cut**: the inspection-outcome-as-label framing; RAG with OpenInference instrumentation and the JSONL sink; deployed public URL with status page and load test; the case-study writeup; the nightly eval workflow.

---

## 17. Conventions for Claude Code

- **Plan first.** For every phase, produce `docs/plans/phase-N.md` with a task list before writing code. Wait for approval.
- **One PR per deliverable.** Don't combine multiple deliverables in one PR.
- **ADRs for non-trivial decisions.** `docs/decisions/NNNN-title.md` with Context / Decision / Consequences.
- **Commit messages**: conventional commits (`feat:`, `fix:`, `chore:`, `docs:`).
- **No TODOs left in code.** If something is deferred, open an issue and link it.
- **No `any` in TypeScript, no `# type: ignore` in Python** — solve the type error.
- **Prefer stdlib and first-party packages** over adding dependencies. Every new dep is documented in the ADR for that phase.
- **Secrets never committed.** `.env.example` only.
- **Every externally-facing error** has a stable error code and a user-readable message.
- **Every ML and eval artifact is versioned** with a timestamp directory and a hash of the training code.
- **When in doubt, flag it.** Do not silently paper over ambiguity.
