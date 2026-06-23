# Case study — NYC Rat Risk Intelligence

*How I built a neighborhood-level rat risk prediction system over NYC open data, deployed it for $0/month, and learned that the label matters more than the model.*

---

## The problem I was actually solving

When I started this project, the obvious dataset to reach for was NYC 311 rodent complaints. There are 229k of them, they're published as a Socrata API, and "predict which neighborhoods will have the most complaints" is a straightforward regression problem.

I didn't do that. Here's why.

311 complaint volume is not a measure of rat infestation. It's a measure of who calls 311. Wealthier neighborhoods with more engaged residents generate more complaints than lower-income neighborhoods — not because they have more rats, but because they have more social capital and less friction with city agencies. A model trained on complaint volume would learn to predict engagement, not infestation, and would send inspectors to neighborhoods that are already over-served.

The better label is in the DOHMH Rodent Inspections dataset: the binary outcome `RESULT == 'Active Rat Signs'`. This is what an inspector found when they showed up, not what a resident reported. It's noisier (inspectors don't cover every block every week) but it's measuring the right thing.

Choosing this label constrained everything downstream — the feature engineering, the evaluation metric, the baseline comparison. And it made the model meaningfully better for the stated use case: proactive inspection routing, not complaint prediction.

---

## Data assembly

Seven data sources, all NYC open data or freely downloadable:

| Source | Join key | Signal |
|---|---|---|
| DOHMH Rodent Inspections | BBL → NTA crosswalk | Label (active rat signs) |
| 311 Rodent Complaints | NTA | Complaint volume (feature, not label) |
| DOHMH Restaurant Inspections | NTA | Pest violation density (04K/04L/08A) |
| DOB Construction Permits | NTA | Disturbance signal (rats flee construction) |
| MapPLUTO | BBL | Building density, land use, lot coverage |
| NOAA Weather (Central Park) | Date | Temperature, precipitation, seasonal cycle |
| Sentinel-2 L2A | Lat/lon → NTA | Vegetation/impervious surface (planned) |

The hardest part was the BBL → NTA crosswalk. BBLs (Borough-Block-Lot codes) are the native key in property data; NTAs are the spatial units for predictions. NYC DCP publishes a crosswalk, but it uses 2010 NTA definitions; the 311 and inspection data now uses 2020 NTA codes. `ml/scripts/tract_crosswalk.py` handles the 2010↔2020 mapping.

The panel is 156 weeks × 223 NTAs = 34,788 rows. After dropping weeks with fewer than 10 inspections in an NTA (too sparse to trust the label), the training set is ~28k rows.

---

## The model that worked

I trained four tabular models:

| Model | Test PR-AUC | Notes |
|---|---|---|
| Logistic Regression | 0.612 | Baseline |
| LightGBM | 0.771 | |
| CatBoost | **0.7947** | Primary production model |
| TabPFN v2 | 0.741 | Per-borough, small-N |

The biggest PR-AUC lifts over the LR baseline came from three feature groups:
1. **Restaurant inspection pest violations** (lagged 4 and 8 weeks) — a strong leading indicator
2. **Spatial lags** (queen-contiguity neighbor NTA features) — rat populations diffuse across borders
3. **Seasonal interaction terms** (temperature × complaint density) — summer months amplify both features

CatBoost beat LightGBM by ~2 points primarily because of better handling of the categorical NTA-id feature. I didn't tune it heavily — the default hyperparameters with depth=8 and 1000 iterations were within 0.3 points of a Bayesian search.

On top of CatBoost, I stacked a Temporal Fusion Transformer (TFT via Darts) trained on 52-week context windows. TFT captures the seasonality and trend components that a static weekly snapshot misses. The fusion meta-learner (isotonic-calibrated logistic regression over out-of-fold predictions) reached PR-AUC 0.7975 on the held-out test period.

---

## What didn't work

**Clay v1.5 satellite embeddings.** The Sentinel-2 rasters ingested fine (~12 GB of imagery). The Clay encoder is a masked autoencoder that produces 768-dimensional patch embeddings from 7-band inputs. The plan was to reduce these to 32 dimensions with PCA and add them as static NTA features.

This didn't run. The Clay inference pass requires ~6 GB of RAM per batch on CPU (the MPS path crashed on macOS on arrays this size). The free-tier Render instance has 512 MB. I documented the cut in ADR 0005.

The silver lining: CatBoost at PR-AUC 0.7947 already beats the spec's 0.78 target (best baseline + 2 points = 0.612 + 0.02 = 0.63 target — wait, that's the LR baseline; the spec means +2 points over the best single baseline, so 0.771 + 0.02 = 0.791; we cleared it). Clay would have pushed the fusion model higher but wasn't load-bearing.

**Chronos-2 fine-tuning.** The challenger temporal model I planned to fine-tune Chronos-2 on the inspection count time series. Chronos (Amazon Research) is a generalist sequence model pre-trained on M-datasets. Fine-tuning on a 156-week × 223-NTA corpus would have been ~32k sequences — feasible, but the default fine-tune requires a GPU. Cut per the spec's cut-line guidance.

---

## RAG over the NYC Health Code

The legal Q&A component retrieves from five NYC Health Code sources:
- NYC Health Code (Article 151 — Rodent Control)
- NYC Administrative Code (Article 3, pest-related sections)
- RCNY Title 24 (Health Commissioner rules)
- DOHMH Enforcement Fact Sheet
- NYC Department of Buildings rodent-adjacent code sections

**Chunking:** Section-aware hierarchical splitting at 800 tokens with 100-token overlap, preserving section headers as chunk metadata. This gives 1,190 chunks total.

**Retrieval:** BM25 (PostgreSQL `tsvector` + `plainto_tsquery`) + dense cosine search (pgvector HNSW, BGE-M3 1024-dim) fused with Reciprocal Rank Fusion (k=60). BGE Reranker v2-M3 selects top-6 chunks for generation.

**Degraded mode:** On Render free tier (512 MB RAM), both BGE-M3 and the Reranker are disabled via env var. BM25-only retrieval provides acceptable precision for well-formed legal queries but lower recall on terminology variations.

**Generation:** Groq (free tier) with a structured prompt that requires the model to cite section numbers from retrieved chunks. Citations are extracted with a regex and surfaced in the UI as linked badges.

---

## Deployment on $0/month

The whole stack runs on free tiers:

- **Render free:** FastAPI + CatBoost model artifact (1.2 MB) + BM25 retrieval
- **Vercel Hobby:** Next.js 16 App Router, MapLibre GL JS choropleth
- **Supabase Free:** 280 MB Postgres with PostGIS, pgvector, pg_trgm; 19,157 pre-materialized predictions

The biggest deployment surprise: Render doesn't set `CWD` to the repository root when running `uv run --package rat-api`. CatBoost artifact paths resolved relative to `os.getcwd()` failed silently (API started in degraded mode with `model_bundle = None`). The fix was `Path(__file__).resolve().parents[4]` as the repo root anchor in `ml/loader.py`.

The second surprise: Render free tier shuts down after 15 minutes of inactivity. A keepalive GitHub Action pings `/health` every 14 minutes to prevent cold starts during business hours.

---

## What I'd do differently

**Start with materialized predictions.** I built the live-inference path first (`/risk/map` calling CatBoost on 223 NTAs per request). It worked locally but timed out in production on the first real request. The materialized-predictions table (`app.risk_predictions`) should have been the first path — compute offline, serve from cache, keep the live path as a fallback only.

**Don't gitignore model artifacts.** The 1.2 MB CatBoost model was in `.gitignore` because I had `ml/artifacts/` globally ignored (to exclude large ONNX models and embedding weights). This meant Render cloned the repo without the model and crashed on startup. The right pattern is to gitignore by extension (`.onnx`, `.pt`, `.npy`) rather than by directory.

**Build the eval suite earlier.** The 50-item gold Q&A set was assembled during Phase 6 hardening. If I'd built it during Phase 2 (RAG setup), I'd have had signal on retrieval quality throughout development instead of discovering the BM25 recall gap at the end.

---

## Results

| Metric | Target | Achieved |
|---|---|---|
| CatBoost test PR-AUC | ≥ 0.791 | **0.7947** |
| Fusion ensemble PR-AUC | — | **0.7975** |
| TFT val_loss | — | **0.163** |
| Top-decile lift | ≥ 1.4× | **1.53×** |
| RAG corpus | ≥ 50 chunks | **1,190 chunks** |
| Monthly infra cost | ≤ $50 | **$0** |

Live demo: [web-beige-three-56.vercel.app](https://web-beige-three-56.vercel.app)
