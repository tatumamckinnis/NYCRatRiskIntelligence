# Phase 2 — Tabular Baseline

**Goal**: a calibrated tabular risk model served from FastAPI, with published metrics and a Dockerized local stack.

**Acceptance check**: `curl localhost:8000/risk/nta/MN2501` returns a valid JSON response with `risk_score`, `risk_decile`, and `top_factors`; test-period PR-AUC ≥ best single-feature baseline by a measurable margin.

---

## Task List

Complexity: **S** (< 1 hr), **M** (1–3 hr), **L** (3–6 hr)

---

### T-14 — Time-series CV + metrics utilities
**Complexity**: S
**Dependencies**: Phase 1 complete
**Files created**:
- `ml/src/rat_ml/eval/__init__.py`
- `ml/src/rat_ml/eval/timeseries_cv.py`
- `ml/src/rat_ml/eval/metrics.py`
- `ml/tests/test_timeseries_cv.py`

**Description**:
- `timeseries_cv.py`: `expanding_window_splits(df, n_folds=5, gap_days=28, test_weeks=4)` — yields `(train_idx, val_idx)` pairs. Train set grows by one fold per iteration; a 28-day gap between train end and val start prevents leakage through reporting lag. Final holdout is the most recent 12 weeks (not yielded as a fold — kept as a separate test set).
- `metrics.py`: `pr_auc(y_true, y_prob)`, `brier_score(y_true, y_prob)`, `top_decile_lift(y_true, y_prob)`, `calibration_summary(y_true, y_prob)` — returns a dict ready to be written to the artifact report.
- Tests: assert no date overlap between any train/val pair; assert gap ≥ 28 days; assert fold count equals `n_folds`.

---

### T-15 — Feature matrix assembly
**Complexity**: S
**Dependencies**: T-14, `features.nta_week_panel` loaded in DB
**Files created**:
- `ml/src/rat_ml/features/feature_matrix.py`

**Description**:
- `load_feature_matrix(db_url) -> pd.DataFrame`: pulls all columns from `features.nta_week_panel` via asyncpg, returns a sorted DataFrame with a `borough` column derived from the NTA prefix (`MN`, `BX`, `BK`, `QN`, `SI`).
- `FEATURE_COLS`: module-level list of all predictor column names (everything except `nta_id`, `week_start`, and the label columns). Clay PCA columns included only when present (detected at runtime).
- `LABEL_COL = "active_rat_signs_ind"`.
- `train_test_split(df, holdout_weeks=12) -> tuple[pd.DataFrame, pd.DataFrame]`: splits chronologically; holdout is the last 12 ISO weeks.
- No DB dependency in tests — unit tests use a synthetic in-memory DataFrame.

---

### T-16 — Tabular model training pipeline
**Complexity**: L
**Dependencies**: T-14, T-15
**Files created**:
- `ml/src/rat_ml/models/__init__.py`
- `ml/src/rat_ml/models/tabular.py`
- `ml/src/rat_ml/models/registry.py`
- `ml/scripts/train_tabular.py`

**Description**:

**`tabular.py`** — three trainers, common interface `train(X_train, y_train, X_val, y_val) -> TrainedModel`:
- `CatBoostTrainer`: `CatBoostClassifier` with `loss_function='Logloss'`, `eval_metric='AUC'`, early stopping on val set, `cat_features` for `borough`.
- `LightGBMTrainer`: `LGBMClassifier`, early stopping.
- `LogisticRegressionTrainer`: sklearn `LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')`.

Each trainer:
1. Runs expanding-window TS-CV (5 folds, 28-day gap) — records fold-by-fold PR-AUC.
2. Retrains on the full train set after CV.
3. Applies isotonic calibration (`CalibratedClassifierCV(method='isotonic')` on the final fold's val set).
4. Computes SHAP values (TreeExplainer for CatBoost/LGB; LinearExplainer for LR) — stores top-20 mean |SHAP| feature importances.
5. Evaluates on the held-out test set: PR-AUC, Brier score, top-decile lift.

**`registry.py`** — `ModelRegistry(artifacts_dir)`:
- `save(model_name, model_obj, metadata: dict) -> str`: writes to `<artifacts_dir>/<model_name>/<timestamp>/`, serialises with `joblib`, writes `registry.json` at the root of `artifacts_dir` pointing to the latest version of each model.
- `load(model_name) -> tuple[Any, dict]`: loads from the path recorded in `registry.json`.

**`train_tabular.py`** — CLI:
```
uv run --package rat-ml --extra ml python ml/scripts/train_tabular.py
```
Trains all three models, writes artifacts, prints ablation table to stdout, writes `report.md` to the artifact dir.

Ablation table columns: `model | cv_pr_auc_mean | cv_pr_auc_std | test_pr_auc | brier | top_decile_lift`.

---

### T-17 — Per-borough TabPFN v2
**Complexity**: M
**Dependencies**: T-16
**Files created**:
- `ml/src/rat_ml/models/tabpfn_borough.py`

**Description**:
- For each borough with ≤ 10,000 training rows, train a `TabPFNClassifier` (v2) on the borough subset.
- `BoroughTabPFNEnsemble`: wraps a dict of `{borough: TabPFNClassifier}`. `predict_proba(X)` routes each row to its borough model; falls back to the main CatBoost model for boroughs without a TabPFN fit.
- Integrated into `train_tabular.py` as a fourth ablation row: `CatBoost + TabPFN`.
- Artifacts saved alongside the other tabular models in the registry.

---

### T-18 — FastAPI app scaffold + config extension
**Complexity**: M
**Dependencies**: T-16 (registry path needed in settings)
**Files created/modified**:
- `api/src/rat_api/main.py` *(new)*
- `api/src/rat_api/config.py` *(add `model_artifacts_dir` field)*
- `api/src/rat_api/models/__init__.py` *(new)*
- `api/src/rat_api/models/risk.py` *(new — Pydantic response schemas)*
- `api/src/rat_api/ml/__init__.py` *(new)*
- `api/src/rat_api/ml/loader.py` *(new)*

**Description**:
- `main.py`: `FastAPI(lifespan=lifespan)` — lifespan loads all tabular models from the registry at startup and stores them in `app.state`. Includes CORS middleware (permissive in dev), Sentry init, and router registration.
- `config.py`: add `model_artifacts_dir: str = "ml/artifacts"` and `model_name: str = "catboost"`.
- `models/risk.py`: Pydantic v2 response models:
  - `RiskFactor(feature: str, contribution: float, direction: Literal["up","down"], readable: str)`
  - `WeekForecast(week: date, risk_score: float, ci_low: float, ci_high: float)`
  - `NtaRiskResponse(nta_id, current_week, risk_score, risk_decile, top_factors, model_version, forecast_12w)`
  - `MapRiskItem(nta_id, risk_score, risk_decile)`
- `ml/loader.py`: `load_models(artifacts_dir, model_name) -> dict` — called in lifespan, returns `{model_name: (model_obj, metadata)}`.

---

### T-19 — Online feature assembly + inference pipeline
**Complexity**: L
**Dependencies**: T-18
**Files created**:
- `api/src/rat_api/ml/features.py`
- `api/src/rat_api/ml/predict.py`

**Description**:
- `features.py`: `async def get_nta_features(nta_id: str, week: date, conn: asyncpg.Connection) -> dict | None` — fetches the single panel row for the given NTA and week. Returns `None` if the row is missing (triggers 503 in the endpoint, not a silent fallback).
- `predict.py`: `predict_risk(model_obj, feature_row: dict, feature_cols: list[str]) -> PredictionResult` — runs the calibrated model, computes SHAP values for the row, returns `risk_score`, `risk_decile` (decile relative to the most recent week's score distribution cached at startup), and `top_factors` (top-5 SHAP contributors with human-readable labels from a hardcoded `FEATURE_LABELS` dict).

---

### T-20 — API endpoints
**Complexity**: M
**Dependencies**: T-19
**Files created**:
- `api/src/rat_api/routes/__init__.py`
- `api/src/rat_api/routes/risk.py`
- `api/src/rat_api/routes/health.py`
- `api/src/rat_api/routes/inspections.py`

**Description**:
- `GET /risk/nta/{nta_id}` → `NtaRiskResponse`. Fetches feature row for the current ISO week; 503 if missing. `forecast_12w` is stubbed with zeroed CI bands in Phase 2 (TFT not yet trained); a `X-Forecast-Stub: true` header flags this.
- `GET /risk/map?week={week}` → `list[MapRiskItem]`. Fetches all NTAs' scores for the given week from `app.risk_predictions` (materialised at training time). Falls back to running inference for all NTAs if the cache table is empty.
- `GET /inspections/nta/{nta_id}?since={date}` → `list[InspectionItem]`. Reads from `raw.rodent_inspections` with a PostGIS bounding-box pre-filter on NTA.
- `GET /health` → `{status, model_version, db_latency_ms, git_sha}`. Returns 200 only if DB ping succeeds and models are loaded; 503 otherwise.

---

### T-21 — Dockerfile + docker-compose.yml
**Complexity**: S
**Dependencies**: T-20
**Files created**:
- `api/Dockerfile`
- `docker-compose.yml` (repo root)

**Description**:
- `Dockerfile`: multi-stage build (builder installs `uv sync --package rat-api`; runtime copies the venv). Non-root user. `CMD ["uvicorn", "rat_api.main:app", "--host", "0.0.0.0", "--port", "8000"]`.
- `docker-compose.yml`: two services — `api` (builds from `api/Dockerfile`, mounts `.env`, exposes `:8000`) and `phoenix` (`arizephoenix/phoenix:latest`, exposes `:6006`). Both services get the `DATABASE_URL` from the host `.env`.

---

### T-22 — API + ML unit tests
**Complexity**: M
**Dependencies**: T-20
**Files created**:
- `api/src/tests/__init__.py`
- `api/src/tests/test_routes_risk.py`
- `api/src/tests/test_routes_health.py`
- `ml/tests/test_tabular.py`
- `ml/tests/test_registry.py`

**Description**:
- `test_routes_risk.py`: route contract tests using `httpx.AsyncClient` with a mocked DB dependency and mocked model. Assert response schema matches `NtaRiskResponse`; assert 503 when feature row is missing.
- `test_routes_health.py`: assert `/health` returns 200 when DB mock pings successfully and models are loaded; assert 503 when DB mock raises.
- `test_tabular.py`: unit tests for `CatBoostTrainer` on a 200-row synthetic DataFrame — assert trained model has a `predict_proba` method; assert calibrated output is in [0, 1]; assert SHAP dict has ≤ 20 keys.
- `test_registry.py`: round-trip test — save a dummy model object, load it back, assert identity.

---

## Task Dependencies (DAG)

```
T-14 (CV + metrics)
  └── T-15 (feature matrix)
        └── T-16 (tabular training)
              ├── T-17 (TabPFN borough)
              └── T-18 (FastAPI scaffold)
                    └── T-19 (inference pipeline)
                          └── T-20 (endpoints)
                                ├── T-21 (Docker)
                                └── T-22 (tests)
```

---

## Files to Create (Complete List)

```
ml/src/rat_ml/eval/__init__.py
ml/src/rat_ml/eval/timeseries_cv.py
ml/src/rat_ml/eval/metrics.py
ml/src/rat_ml/models/__init__.py
ml/src/rat_ml/models/tabular.py
ml/src/rat_ml/models/registry.py
ml/src/rat_ml/models/tabpfn_borough.py
ml/src/rat_ml/features/feature_matrix.py
ml/scripts/train_tabular.py
ml/tests/test_timeseries_cv.py
ml/tests/test_tabular.py
ml/tests/test_registry.py
api/src/rat_api/main.py
api/src/rat_api/models/__init__.py
api/src/rat_api/models/risk.py
api/src/rat_api/ml/__init__.py
api/src/rat_api/ml/loader.py
api/src/rat_api/ml/features.py
api/src/rat_api/ml/predict.py
api/src/rat_api/routes/__init__.py
api/src/rat_api/routes/risk.py
api/src/rat_api/routes/health.py
api/src/rat_api/routes/inspections.py
api/src/tests/__init__.py
api/src/tests/test_routes_risk.py
api/src/tests/test_routes_health.py
api/Dockerfile
docker-compose.yml
```

**Modified**:
- `api/src/rat_api/config.py` — add `model_artifacts_dir`, `model_name`

---

## Open Questions

1. **Model artifacts location**: the API container needs access to `ml/artifacts/`. In local dev this is a bind mount; in production (Render) we'll need to bundle the artifacts into the Docker image or pull from object storage. For Phase 2, the Dockerfile will `COPY ml/artifacts/ /app/ml/artifacts/` — this means re-building the image when a new model is trained. Acceptable for now; Phase 3 will introduce a proper artifact pull step.

2. **`app.risk_predictions` pre-materialisation**: `GET /risk/map` is supposed to read from the cache table. For Phase 2 we add a `materialize_predictions.py` script that runs inference for all NTAs for the current week and writes to `app.risk_predictions`. The endpoint falls back to live inference if the table is empty.

3. **SHAP for LR**: `LinearExplainer` requires a background dataset; we'll use the training set mean as the background. This is an approximation but sufficient for top-factor display.

4. **Human-readable feature labels**: a hardcoded `FEATURE_LABELS` dict in `predict.py` maps column names to display strings (e.g. `"complaints_lag_4w"` → `"311 complaints 4 weeks ago"`). These will be iterated on in Phase 5 when the frontend renders them.
