"""Unit tests for tabular model trainers (T-16).

All tests use a small synthetic DataFrame — no DB or GPU required.
CatBoost and LightGBM are imported lazily; tests are skipped if the
extra deps aren't installed.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from rat_ml.eval.timeseries_cv import holdout_split


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

def _make_training_data(n_weeks: int = 80, n_ntas: int = 10, seed: int = 0) -> pd.DataFrame:
    """Minimal synthetic NTA-week DataFrame with all feature columns."""
    rng = np.random.default_rng(seed)
    weeks = [date(2021, 1, 4) + timedelta(weeks=i) for i in range(n_weeks)]
    boroughs = ["MN", "BX", "BK", "QN", "SI"]

    rows = []
    for i, w in enumerate(weeks):
        for nta in range(n_ntas):
            borough = boroughs[nta % len(boroughs)]
            label = int(rng.random() > 0.7)
            rows.append(
                {
                    "nta_id": f"{borough}{nta:04d}",
                    "week_start": w,
                    "borough": borough,
                    "complaints_count": int(rng.integers(0, 20)),
                    "complaints_lag_1w": int(rng.integers(0, 20)),
                    "complaints_lag_4w": int(rng.integers(0, 20)),
                    "complaints_lag_12w": int(rng.integers(0, 20)),
                    "rest_pest_violations_count": int(rng.integers(0, 5)),
                    "permits_active_count": int(rng.integers(0, 10)),
                    "demolitions_count": int(rng.integers(0, 2)),
                    "weather_tavg_c": float(rng.uniform(-5, 30)),
                    "weather_prcp_mm": float(rng.uniform(0, 50)),
                    "weather_hdd": float(rng.uniform(0, 20)),
                    "weather_cdd": float(rng.uniform(0, 20)),
                    "units_total": int(rng.integers(100, 5000)),
                    "year_built_median": int(rng.integers(1900, 2020)),
                    "landuse_residential_pct": float(rng.uniform(0, 1)),
                    "landuse_commercial_pct": float(rng.uniform(0, 0.3)),
                    "neighbor_active_rat_signs_rate_lag_1w": float(rng.uniform(0, 1)),
                    "neighbor_complaints_count_lag_4w": float(rng.uniform(0, 20)),
                    "regime_covid": int(i < 10),
                    "regime_8pm_setout": int(i > 40),
                    "regime_commercial_containerization": int(i > 50),
                    "regime_residential_containerization": int(i > 60),
                    "regime_rmz_active": 0,
                    "active_rat_signs_ind": label,
                }
            )

    return pd.DataFrame(rows)


FEATURE_COLS = [
    "borough",
    "complaints_count", "complaints_lag_1w", "complaints_lag_4w", "complaints_lag_12w",
    "rest_pest_violations_count", "permits_active_count", "demolitions_count",
    "weather_tavg_c", "weather_prcp_mm", "weather_hdd", "weather_cdd",
    "units_total", "year_built_median",
    "landuse_residential_pct", "landuse_commercial_pct",
    "neighbor_active_rat_signs_rate_lag_1w", "neighbor_complaints_count_lag_4w",
    "regime_covid", "regime_8pm_setout", "regime_commercial_containerization",
    "regime_residential_containerization", "regime_rmz_active",
]
LABEL_COL = "active_rat_signs_ind"


@pytest.fixture(scope="module")
def datasets():
    df = _make_training_data()
    df["week_start"] = pd.to_datetime(df["week_start"])
    train, test = holdout_split(df, holdout_weeks=8)
    return train, test


# ---------------------------------------------------------------------------
# LR (always available — no extra install required)
# ---------------------------------------------------------------------------

def test_lr_trains_and_predicts(datasets) -> None:
    from rat_ml.models.tabular import LRTrainer  # noqa: PLC0415

    train, test = datasets
    result = LRTrainer().fit(train, test, FEATURE_COLS, n_folds=3)

    assert result.model is not None
    assert result.cv_pr_auc_mean > 0
    assert set(result.test_metrics.keys()) == {"pr_auc", "roc_auc", "brier", "top_decile_lift"}


def test_lr_calibrated_output_in_range(datasets) -> None:
    from rat_ml.models.tabular import LRTrainer, _encode_categoricals  # noqa: PLC0415

    train, test = datasets
    result = LRTrainer().fit(train, test, FEATURE_COLS, n_folds=3)

    X_test, _, _ = _encode_categoricals(
        test[FEATURE_COLS].copy(),
        test[FEATURE_COLS].iloc[:1].copy(),
        ["borough"],
    )
    probs = result.model.predict_proba(X_test)[:, 1]
    assert (probs >= 0).all() and (probs <= 1).all()


def test_lr_fold_count(datasets) -> None:
    from rat_ml.models.tabular import LRTrainer  # noqa: PLC0415

    train, test = datasets
    result = LRTrainer().fit(train, test, FEATURE_COLS, n_folds=3)
    assert len(result.fold_metrics) == 3


def test_lr_feature_cols_preserved(datasets) -> None:
    from rat_ml.models.tabular import LRTrainer  # noqa: PLC0415

    train, test = datasets
    result = LRTrainer().fit(train, test, FEATURE_COLS, n_folds=3)
    assert result.feature_cols == FEATURE_COLS


# ---------------------------------------------------------------------------
# CatBoost (skipped if not installed)
# ---------------------------------------------------------------------------

catboost = pytest.importorskip("catboost", reason="catboost not installed")


def test_catboost_trains(datasets) -> None:
    from rat_ml.models.tabular import CatBoostTrainer  # noqa: PLC0415

    train, test = datasets
    result = CatBoostTrainer(iterations=50).fit(train, test, FEATURE_COLS, n_folds=3)
    assert result.model is not None
    assert result.cv_pr_auc_mean > 0


def test_catboost_calibrated_output_in_range(datasets) -> None:
    from rat_ml.models.tabular import CatBoostTrainer  # noqa: PLC0415

    train, test = datasets
    result = CatBoostTrainer(iterations=50).fit(train, test, FEATURE_COLS, n_folds=3)
    probs = result.model.predict_proba(test[FEATURE_COLS])[:, 1]
    assert (probs >= 0).all() and (probs <= 1).all()


# ---------------------------------------------------------------------------
# Registry round-trip with a trained LR model
# ---------------------------------------------------------------------------

def test_registry_round_trip_with_trained_model(datasets, tmp_path) -> None:
    from rat_ml.models.registry import ModelRegistry  # noqa: PLC0415
    from rat_ml.models.tabular import LRTrainer, _encode_categoricals  # noqa: PLC0415

    train, test = datasets
    result = LRTrainer().fit(train, test, FEATURE_COLS, n_folds=3)

    registry = ModelRegistry(tmp_path / "artifacts")
    registry.save("lr_test", result.model, metadata={"test_pr_auc": result.test_metrics["pr_auc"]})

    loaded_model, meta = registry.load("lr_test")
    assert meta["test_pr_auc"] == pytest.approx(result.test_metrics["pr_auc"])

    X_test, _, _ = _encode_categoricals(
        test[FEATURE_COLS].copy(),
        test[FEATURE_COLS].iloc[:1].copy(),
        ["borough"],
    )
    probs = loaded_model.predict_proba(X_test)[:, 1]
    assert len(probs) == len(test)
