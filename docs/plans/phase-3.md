# Phase 3 — Multi-Modal Ensemble

**Goal**: TFT + Chronos-2 + Clay v1.5 embeddings integrated via late-fusion stacked meta-learner; `/risk/nta/{nta_id}` returns real 12-week forecasts with CI bands.

**Acceptance check**: full ensemble PR-AUC ≥ tabular-only + 2 points; `forecast_12w` in the API response contains non-stub values with `ci_low < risk_score < ci_high`.

---

## Status: what's already done

From the `feat(T-23..T-29)` commit:
- `ml/src/rat_ml/features/tft_dataset.py` — Darts TimeSeries builder ✅
- `ml/src/rat_ml/models/tft_trainer.py` — full TFT training with quantile regression and early stopping ✅
- `ml/scripts/train_tft.py` — CLI entry point ✅
- `ml/scripts/materialize_tft_forecasts.py` — forecast materialization script ✅
- `ml/src/rat_ml/models/tabpfn_borough.py` — per-borough TabPFN v2 ✅

**Still to build**: Sentinel-2 ingest, Clay embeddings, Chronos-2 fine-tune, fusion model, API wiring, ablation table, architecture diagram.

---

## Task List

Complexity: **S** (< 1 hr), **M** (1–3 hr), **L** (3–6 hr)

---

### T-30 — Sentinel-2 ingest via MPC STAC
**Complexity**: M
**Files created**:
- `ml/src/rat_ml/data/ingest_sentinel2.py`

**Description**:
- Use `pystac_client` + `planetary_computer` to search for Sentinel-2 L2A scenes over NYC tiles `18TWL` and `18TWK`.
- For each NTA, build a quarterly cloud-masked mosaic (median composite of cloud-free pixels) using `stackstac`.
- Output: a per-NTA quarterly raster clipped to the NTA bounding box, stored locally in `data/sentinel2/<nta_id>/<quarter>/` (gitignored).
- Idempotent: skip NTAs that already have a raster for the current quarter.
- `PLANETARY_COMPUTER_KEY` env var is optional; most Sentinel-2 endpoints are anonymous.

---

### T-31 — Clay v1.5 embeddings + PCA
**Complexity**: L
**Files created**:
- `ml/src/rat_ml/features/clay_embeddings.py`

**Description**:
- Load frozen Clay v1.5 model weights (HuggingFace `made-with-clay/Clay`).
- For each NTA raster from T-30, run the frozen encoder to produce a high-dimensional embedding.
- Apply PCA (fit on training NTAs, transform all) to reduce to 32 dimensions.
- Write `clay_pca_0` through `clay_pca_31` columns into `features.nta_week_panel` via an UPDATE (join on NTA id; values repeat across all weeks for the same NTA since the raster is quarterly).
- Save PCA model to `ml/artifacts/clay_pca.joblib`.
- Migration: `ALTER TABLE features.nta_week_panel ADD COLUMN IF NOT EXISTS clay_pca_N NUMERIC` for N in 0..31 (in a new Alembic migration).

---

### T-32 — Chronos-2 fine-tune
**Complexity**: L
**Files created**:
- `ml/src/rat_ml/models/chronos_trainer.py`
- `ml/scripts/train_chronos.py`

**Description**:
- Fine-tune `amazon/chronos-t5-small` (Chronos-2) on the NTA-week `active_rat_signs_count` series.
- Use HuggingFace `transformers` + `torch` for fine-tuning; 12-week forecast horizon matching TFT.
- Quantile output at p10/p50/p90 via Monte Carlo sampling (100 draws).
- Save fine-tuned weights to `ml/artifacts/tft_checkpoints/chronos/` via `ModelRegistry`.
- Document in training log why Chronos-2 is used as a challenger (zero-shot generalization + different inductive bias to TFT).

---

### T-33 — Train TFT (run existing code)
**Complexity**: S
**Files modified**: none (code already complete)

**Description**:
- Run `ml/scripts/train_tft.py` with `--accelerator auto` (uses MPS on Apple Silicon, CPU fallback).
- Record val_loss and n_series in the ablation table.
- Commit the artifact pointer in `registry.json`.

---

### T-34 — Fusion model: stacked meta-learner
**Complexity**: L
**Files created**:
- `ml/src/rat_ml/models/fusion.py`
- `ml/scripts/train_fusion.py`

**Description**:
- Assemble out-of-fold (OOF) predictions from:
  - CatBoost (already computed during `train_tabular.py` CV)
  - TFT p50 for the current week (derived from forecast)
  - Chronos-2 p50 for the current week
  - Clay PCA features (32 columns, repeated per week)
- Meta-learner: `LogisticRegression(C=1.0)` (primary); shallow MLP (ablation row).
- Apply isotonic calibration on the meta-learner output.
- Final artifact: a single callable `FusionModel.predict_proba(X_row)` that returns calibrated probability + top-N SHAP factors.
- Save to registry under name `"fusion"`.
- Ablation table rows: CatBoost alone / +TFT / +Clay / +Chronos-2 / full ensemble.

---

### T-35 — Wire `forecast_12w` in API with real TFT CI bands
**Complexity**: M
**Files modified**:
- `api/src/rat_api/ml/loader.py` — load TFT model in lifespan alongside tabular model
- `api/src/rat_api/ml/predict.py` — call `forecast_nta()` to produce 12-week CI bands
- `api/src/rat_api/routes/risk.py` — remove `X-Forecast-Stub: true` header; populate `forecast_12w`

**Description**:
- Load the TFT model at lifespan startup (stored in `app.state.tft_model`).
- In the `/risk/nta/{nta_id}` handler, call `forecast_nta()` to produce `{week, p10, p50, p90}` for 12 weeks.
- Map `p10 → ci_low`, `p50 → risk_score`, `p90 → ci_high` in the `WeekForecast` response schema.
- Graceful fallback: if TFT model is not loaded, return the stub (zeroed CI) with the `X-Forecast-Stub: true` header still set — never 500.

---

### T-36 — Extended ablation table + architecture diagram
**Complexity**: S
**Files modified/created**:
- `ml/artifacts/tabular/ablation.md` — add TFT, Clay, Chronos, ensemble rows
- `docs/architecture.md` — rendered architecture diagram

**Description**:
- Add rows to the ablation table: CatBoost / +TFT / +Clay / +Chronos-2 / full ensemble (PR-AUC, Brier, Top-decile lift).
- Render architecture diagram using Mermaid (committed as source + PNG) showing the data flow from raw sources → features → models → fusion → API → frontend.
- Commit both.

---

## Task Dependencies (DAG)

```
T-30 (Sentinel-2 ingest)
  └── T-31 (Clay embeddings)
        └── T-34 (Fusion model)
              ├── T-36 (Ablation + diagram)

T-33 (Train TFT)
  └── T-34 (Fusion model)
        └── T-35 (Wire API forecast)

T-32 (Chronos-2 fine-tune)
  └── T-34 (Fusion model)
```

---

## Files to Create (Complete List)

```
ml/src/rat_ml/data/ingest_sentinel2.py
ml/src/rat_ml/features/clay_embeddings.py
ml/src/rat_ml/models/chronos_trainer.py
ml/src/rat_ml/models/fusion.py
ml/scripts/train_chronos.py
ml/scripts/train_fusion.py
docs/architecture.md (update)
```

**Modified**:
- `api/src/rat_api/ml/loader.py` — load TFT at startup
- `api/src/rat_api/ml/predict.py` — call TFT forecast
- `api/src/rat_api/routes/risk.py` — real forecast_12w response

**New migration**:
- `api/alembic/versions/<hash>_add_clay_pca_columns.py` — add `clay_pca_0..31` to `features.nta_week_panel`

---

## Open Questions

1. **Clay model weights**: `made-with-clay/Clay` on HuggingFace. Confirm the correct model ID and whether it requires accepting a license before use.
2. **MPS vs CPU for TFT**: Apple Silicon MPS is supported by PyTorch Lightning via `--accelerator mps`. TFT training on CPU with 223 NTAs × 52 weeks context will be slow (~2–4 hrs). MPS should be ~5× faster.
3. **Chronos-2 fine-tune data size**: 223 NTAs × ~150 weekly observations = ~33k time steps. Small enough for a fine-tune on CPU/MPS in a reasonable time.
4. **OOF predictions for fusion**: CatBoost OOF predictions are already saved in the tabular artifact. TFT and Chronos OOF predictions need to be produced during their respective training runs and written to the artifact directory for the fusion script to consume.
