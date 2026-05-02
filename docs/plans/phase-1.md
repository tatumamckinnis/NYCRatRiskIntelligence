# Phase 1 — Data Foundation

**Goal**: a materialized `features.nta_week_panel` table with properly joined keys, a reporting-bias analysis notebook, and a data-quality report.

**Acceptance check**: `uv run python -m ml.scripts.build_panel` produces `features.nta_week_panel` with >100k rows and no NULLs in required columns.

---

## Task List

Tasks are ordered by dependency. Complexity ratings: **S** (< 1 hr), **M** (1–3 hr), **L** (3–6 hr), **XL** (> 6 hr).

---

### T-01 — Monorepo scaffold and tooling
**Complexity**: M
**Dependencies**: none
**Files created**:
- `pyproject.toml` (workspace root)
- `pnpm-workspace.yaml`
- `api/pyproject.toml`
- `ml/pyproject.toml`
- `evals/pyproject.toml`
- `.env.example`
- `.gitignore`
- `README.md` (stub)

**Description**: Initialize `uv` workspace with `api`, `ml`, and `evals` as members. Pin Python 3.12. Add `ruff`, `mypy`, and `pytest` as dev deps in each package. Confirm `uv sync --all-packages` resolves cleanly. Add `.gitignore` entries for `.env`, `data/`, `ml/artifacts/`, `evals/results/`.

**ADR candidate**: none (straight from spec).

---

### T-02 — Supabase project setup and initial migration
**Complexity**: M
**Dependencies**: T-01 + Supabase Free project provisioned by user
**Files created**:
- `api/src/rat_api/db.py` (async SQLAlchemy + asyncpg engine factory)
- `api/src/rat_api/config.py` (Pydantic Settings, reads `DATABASE_URL`, `DIRECT_DATABASE_URL`)
- `api/alembic.ini`
- `api/alembic/env.py`
- `api/alembic/versions/0001_initial_schemas.py`

**Description**: Enable PostGIS, pgvector, pg_trgm, uuid-ossp via the Alembic migration (using `CREATE EXTENSION IF NOT EXISTS`). Create the three schemas (`raw`, `features`, `app`). Full DDL for all Section 5.2 tables:
- `raw.rodent_inspections`
- `raw.complaints_nta_week` (note: per spec amendment, 311 is aggregated at ingest — **not** per-complaint)
- `raw.restaurant_inspections`
- `raw.dob_permits`
- `raw.weather_daily`
- `features.nta_week_panel` (all columns including 32 Clay PCA columns — see ADR below)
- `app.health_code_chunks`
- `app.chat_sessions`
- `app.chat_messages`
- `app.risk_predictions`

**ADR candidate**: `docs/decisions/0002-clay-pca-columns.md` — whether to use 32 separate `NUMERIC` columns (spec default) vs. a single `VECTOR(32)` column. Recommendation: use `VECTOR(32)` for cleaner DDL and native pgvector compatibility; document the deviation.

**Note**: The `app.health_code_chunks` table (for Phase 4 RAG) is created here so Phase 4 can populate it without a schema migration. Same for `app.risk_predictions` (Phase 2).

---

### T-03 — NTA boundary data and crosswalk table
**Complexity**: M
**Dependencies**: T-02
**Files created**:
- `ml/src/data/tract_crosswalk.py`
- `ml/scripts/load_nta_boundaries.py`
- `data/nta_2020_boundaries.geojson` (downloaded, gitignored)
- `api/alembic/versions/0002_nta_reference.py`

**Description**:
- Download NTA 2020 boundary GeoJSON from NYC DCP (direct URL, no API key).
- Load into a `raw.nta_boundaries (nta_id TEXT PK, nta_name TEXT, borough TEXT, geom GEOMETRY(MultiPolygon, 4326), area_sq_m NUMERIC)` reference table.
- Implement `tract_crosswalk.py`: loads the 2010→2020 NTA crosswalk (available as CSV from NYC DCP) and provides an `allocate(value, source_nta_2010, weight_col)` function using area-weighted allocation.
- Write tests: all 2020 NTA IDs present; allocation weights sum to 1.0 per source NTA.

**ADR candidate**: none.

---

### T-04 — BBL join utility
**Complexity**: M
**Dependencies**: T-02
**Files created**:
- `ml/src/data/bbl_join.py`
- `ml/tests/test_bbl_join.py`

**Description**: Implement `normalize_bbl(raw: str) -> str` that zero-pads to 10 chars. Implement `resolve_bbl(bbl, appbbl)` that returns `appbbl` when `bbl != appbbl` (condo billing-BBL logic from PLUTO). Implement `emit_unmatched_report(source: str, total: int, unmatched: int)` that writes a row to the data-quality report. Write unit tests for edge cases: NULL BBL, non-numeric BBL, mismatched length.

**ADR candidate**: none.

---

### T-05 — Ingest: DOHMH Rodent Inspections
**Complexity**: M
**Dependencies**: T-02, T-04, NYC Socrata token in env
**Files created**:
- `ml/src/data/ingest_rodent_inspections.py`
- `ml/tests/test_ingest_rodent_inspections.py`

**Description**:
- Use `sodapy.Socrata` with `$limit=50000`, `$offset` pagination, Socrata ID `p937-wjvj` (fallback `jh4g-rp64`).
- **Cap to last 3 years** (`inspection_date >= NOW() - INTERVAL '3 years'`) at the Socrata query level via `$where`.
- Cast BBL to 10-char via `normalize_bbl`. Emit geometry from lat/lon if present; set projection EPSG:4326.
- Upsert into `raw.rodent_inspections` on `inspection_id`.
- Emit OpenTelemetry span per batch with `batch.size` and `batch.offset` attributes.
- **Idempotency test**: run twice, assert row count unchanged.
- **Geom bounding-box test**: assert all non-NULL `geom` values fall within NYC bounding box `(-74.26, 40.49, -73.68, 40.92)`.

---

### T-06 — Ingest: 311 rodent complaints (NTA-week aggregation)
**Complexity**: M
**Dependencies**: T-02, T-03 (NTA boundaries needed for spatial join)
**Files created**:
- `ml/src/data/ingest_311.py`
- `ml/tests/test_ingest_311.py`

**Description**:
- Fetch `complaint_type='Rodent'` from Socrata `erm2-nwe9`.
- Spatial-join each complaint point to NTA 2020 polygons using `ST_Within` or `ST_Contains` (PostGIS).
- Truncate to `(nta_id, week_start, complaint_count)` — do **not** store per-complaint rows.
- Upsert into `raw.complaints_nta_week` on `(nta_id, week_start)`.
- Incremental update: cursor on `updated_date`, store last cursor in a `raw.ingest_cursors (source TEXT PK, cursor_value TEXT)` table.

---

### T-07 — Ingest: Restaurant inspections, DOB permits, PLUTO, weather
**Complexity**: L
**Dependencies**: T-02, T-04
**Files created**:
- `ml/src/data/ingest_restaurant_inspections.py`
- `ml/src/data/ingest_dob_permits.py`
- `ml/src/data/ingest_pluto.py`
- `ml/src/data/ingest_weather.py`

**Description** (one script each):

- **Restaurant**: Socrata `43nn-pn8j`. Flag violation codes `04K`, `04L`, `08A`. Upsert on inspection serial number.
- **DOB permits**: Socrata `ipu4-2q9a` + `rbx6-tga4`. Track `issuance_date`, `job_type`, `work_type`. Incremental on `issuance_date`. Normalize BBL.
- **PLUTO**: Download from DCP direct URL (latest `25v4+` vintage). Load per-BBL lot file. Normalize BBL. Key fields: `unitsres`, `unitstotal`, `yearbuilt`, `landuse`, `bldgclass`, `appbbl`. Upsert on BBL. ~860k rows — load once per quarter.
- **Weather**: `meteostat` Python library, station `USW00094728` (Central Park). Daily fetch of `tavg`, `tmin`, `tmax`, `prcp`, `snow`. Compute HDD (base 18°C) and CDD (base 18°C). Upsert on `date`.

---

### T-08 — Feature engineering: NTA-week panel assembly
**Complexity**: L
**Dependencies**: T-05, T-06, T-07, T-03
**Files created**:
- `ml/scripts/build_panel.py` (the acceptance-check entry point)
- `ml/src/features/temporal.py`

**Description**:
- Join all `raw.*` tables at NTA-week grain.
- Label columns: `active_rat_signs_count`, `inspections_count`, `active_rat_signs_rate`, `active_rat_signs_ind`.
- 311 lag features: `complaints_lag_1w`, `complaints_lag_4w`, `complaints_lag_12w` (window functions over `raw.complaints_nta_week`).
- Restaurant pest violations count per NTA-week (join via PostGIS BBL→NTA crosswalk).
- DOB permits: active permits count, demolitions count per NTA-week.
- Weather: join on `week_start` date (use Monday of the week).
- PLUTO aggregates: `SUM(unitsres)`, `MEDIAN(yearbuilt)`, `landuse` percentages — static per NTA (join once, not per-week).
- Write to `features.nta_week_panel` via `INSERT … ON CONFLICT DO UPDATE`.
- `temporal.py` exposes the lag-window SQL as composable fragments.

---

### T-09 — Feature engineering: spatial lags
**Complexity**: M
**Dependencies**: T-08, T-03 (NTA boundaries)
**Files created**:
- `ml/src/features/spatial_lags.py`

**Description**:
- Build queen-contiguity adjacency matrix from `raw.nta_boundaries` using `libpysal.weights.Queen` on the GeoDataFrame.
- For each NTA-week, compute `neighbor_active_rat_signs_rate_lag_1w` and `neighbor_complaints_count_lag_4w` as the mean across all contiguous NTAs (using the lag columns already in the panel).
- Update `features.nta_week_panel` in place.
- Test: verify that Manhattan NTAs (island) have no cross-borough neighbors and that adjacency matrix is symmetric.

---

### T-10 — Feature engineering: regime indicators
**Complexity**: S
**Dependencies**: T-08
**Files created**:
- `ml/src/features/regime_indicators.py`

**Description**: Update `features.nta_week_panel` to populate the five boolean regime columns using hardcoded date ranges:
- `regime_covid`: `week_start BETWEEN '2020-03-01' AND '2020-06-30'`
- `regime_8pm_setout`: `week_start >= '2023-04-01'`
- `regime_commercial_containerization`: `week_start >= '2023-07-31'` (tightened `'2024-03-01'` — both threshold rows remain, use the later date as full implementation)
- `regime_residential_containerization`: `week_start >= '2024-11-01'` (simplified; zone-level variation deferred to Phase 3 enhancement)
- `regime_rmz_active`: `FALSE` for all rows in Phase 1 (RMZ data not yet ingested; will be updated when available)

Test: spot-check a row from March 2020 and a row from November 2024.

---

### T-11 — Data quality report
**Complexity**: S
**Dependencies**: T-08
**Files created**:
- `ml/scripts/data_quality_report.py`
- `ml/artifacts/data_quality/<date>.md` (generated, gitignored)

**Description**: Generate a markdown report with:
- Row counts per `raw.*` table
- Unmatched BBL counts per source (from T-04's `emit_unmatched_report`)
- NULL rates for required `features.nta_week_panel` columns
- Date range coverage per source
- `features.nta_week_panel` total row count and NTA count

Run as part of `build_panel.py` at the end.

---

### T-12 — Reporting bias notebook
**Complexity**: M
**Dependencies**: T-08
**Files created**:
- `ml/notebooks/01_reporting_bias.ipynb`

**Description**: Exploratory notebook (not production code) showing:
- 311 complaint density by NTA vs. inspection-outcome `Active Rat Signs` rate, grouped by NTA median income decile (use ACS data via Census API or a pre-downloaded CSV).
- Scatterplot: complaint rate vs. inspection rate, colored by income decile.
- Key finding: whether high-complaint NTAs are inspected at higher rates independent of actual infestation, and whether the model's label (inspection outcome) inherits this bias.
- Conclude with 2–3 bullet "implications for model users" — this feeds directly into the `/about` page and the case study.

**ADR candidate**: `docs/decisions/0003-reporting-bias.md` — documenting the bias finding and what we do (and don't) do about it.

---

### T-13 — Test suite for Phase 1
**Complexity**: M
**Dependencies**: T-02 through T-11
**Files created**:
- `ml/tests/test_geom_bbox.py`
- `ml/tests/test_crosswalk.py`
- `ml/tests/test_panel_nulls.py`
- `ml/tests/test_idempotency.py` (combined for all ingest scripts)

**Description**: Required tests per Section 10.1:
- **Geom bounding-box**: `ST_X(geom) BETWEEN -74.26 AND -73.68 AND ST_Y(geom) BETWEEN 40.49 AND 40.92` for all non-NULL geoms in `raw.rodent_inspections`.
- **Crosswalk sanity**: 2010→2020 allocation weights sum to 1.0 per source NTA; all 2020 NTA codes from `raw.nta_boundaries` appear.
- **Panel nulls**: `active_rat_signs_count`, `inspections_count`, `active_rat_signs_ind`, `nta_id`, `week_start` are never NULL.
- **Idempotency**: run each ingest script twice; row count must be identical on the second run.

Tests run against a live (dev) Supabase database. Mark them `pytest.mark.integration` so they can be skipped in CI without a DB connection.

---

## Task Dependencies (DAG)

```
T-01 (scaffold)
  └── T-02 (migrations)
        ├── T-03 (NTA boundaries)
        │     ├── T-06 (311 ingest)
        │     └── T-09 (spatial lags)
        ├── T-04 (BBL join util)
        │     ├── T-05 (rodent ingest)
        │     └── T-07 (restaurant, DOB, PLUTO, weather)
        └── T-02 also enables T-08 (panel assembly)
              ← requires T-05, T-06, T-07 complete
              └── T-09 (spatial lags)
              └── T-10 (regime indicators)
              └── T-11 (data quality report)
              └── T-12 (bias notebook)
              └── T-13 (test suite)
```

Parallelizable once T-02 and T-04 are done: **T-05**, **T-06**, **T-07** can all run concurrently.

---

## Files to Create (Complete List)

```
pyproject.toml
pnpm-workspace.yaml
.env.example
.gitignore
README.md
api/pyproject.toml
api/alembic.ini
api/alembic/env.py
api/alembic/versions/0001_initial_schemas.py
api/alembic/versions/0002_nta_reference.py
api/src/rat_api/__init__.py
api/src/rat_api/config.py
api/src/rat_api/db.py
ml/pyproject.toml
ml/scripts/build_panel.py
ml/scripts/load_nta_boundaries.py
ml/scripts/data_quality_report.py
ml/src/__init__.py
ml/src/data/__init__.py
ml/src/data/ingest_rodent_inspections.py
ml/src/data/ingest_311.py
ml/src/data/ingest_restaurant_inspections.py
ml/src/data/ingest_dob_permits.py
ml/src/data/ingest_pluto.py
ml/src/data/ingest_weather.py
ml/src/data/bbl_join.py
ml/src/data/tract_crosswalk.py
ml/src/features/__init__.py
ml/src/features/temporal.py
ml/src/features/spatial_lags.py
ml/src/features/regime_indicators.py
ml/tests/__init__.py
ml/tests/test_bbl_join.py
ml/tests/test_geom_bbox.py
ml/tests/test_crosswalk.py
ml/tests/test_panel_nulls.py
ml/tests/test_idempotency.py
ml/tests/test_ingest_rodent_inspections.py
ml/tests/test_ingest_311.py
ml/notebooks/01_reporting_bias.ipynb
evals/pyproject.toml
docs/decisions/0001-single-cloud.md   ← stub (full write in Phase 6)
docs/decisions/0002-clay-pca-columns.md
docs/decisions/0003-reporting-bias.md ← written after notebook finding
```

---

## ADRs to Write in Phase 1

| File | Decision |
|---|---|
| `docs/decisions/0001-single-cloud.md` | Deliberate choice to deploy to a single cloud (Render + Vercel + Supabase Free) rather than multi-cloud; Docker image is portable |
| `docs/decisions/0002-clay-pca-columns.md` | Use `VECTOR(32)` for Clay PCA embeddings in `features.nta_week_panel` instead of 32 separate `NUMERIC` columns |
| `docs/decisions/0003-reporting-bias.md` | Finding from the bias notebook: what bias exists in the inspection-outcome label and what we communicate to users |

---

## Open Questions (Resolve Before Starting)

1. **Supabase project provisioned?** Need `DATABASE_URL` and `DIRECT_DATABASE_URL` in env before T-02.
2. **NYC Socrata token?** Need `NYC_SOCRATA_APP_TOKEN` before T-05 through T-07.
3. **Clay PCA column format**: approve `VECTOR(32)` instead of 32 separate `NUMERIC` columns? (see ADR-0002 above)
4. **ACS income data for bias notebook (T-12)**: use a pre-downloaded CSV from NYC Planning (simplest) or Census API? Recommend the pre-downloaded CSV to avoid another API key dependency.
5. **`raw.ingest_cursors` table**: not in the Section 5.2 DDL — should it be added to the initial migration or created in T-06? Recommend adding to `0001_initial_schemas.py` since it's needed by any incremental ingest script.
