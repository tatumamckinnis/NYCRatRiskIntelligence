"""Stacked meta-learner fusion model (T-34).

Combines out-of-fold (OOF) predictions from:
  - CatBoost tabular model
  - TFT p50 forecast (current-week prediction)
  - Chronos-2 p50 forecast (current-week prediction)
  - Clay PCA features (32 dims, static per NTA)

Meta-learner: LogisticRegression (primary), shallow MLP (ablation).
Isotonic calibration is applied on top of the meta-learner output.

The final FusionModel exposes a sklearn-compatible interface:
  predict_proba(X) -> array of shape (N, 2)

OOF prediction files are expected at:
  ml/artifacts/tabular/catboost/<latest>/oof_predictions.json
  ml/artifacts/tft/<latest>/oof_predictions.json
  ml/artifacts/chronos/<latest>/oof_predictions.json

Each OOF file is a JSON mapping {nta_id|week_start: predicted_prob}.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

N_CLAY_PCA = 32

# Column names in the meta-feature matrix
CATBOOST_COL = "catboost_oof"
TFT_COL = "tft_oof_p50"
CHRONOS_COL = "chronos_oof_p50"
CLAY_COLS = [f"clay_pca_{i}" for i in range(N_CLAY_PCA)]

META_FEATURE_COLS = [CATBOOST_COL, TFT_COL, CHRONOS_COL] + CLAY_COLS
LABEL_COL = "active_rat_signs_ind"


# ---------------------------------------------------------------------------
# OOF loading helpers
# ---------------------------------------------------------------------------

def _load_oof(artifact_path: Path) -> dict[str, float]:
    """Load an OOF prediction JSON file.

    Expected format: {"<nta_id>|<week_start>": <prob>, ...}
    """
    with open(artifact_path) as f:
        return json.load(f)


def _find_latest_artifact(artifacts_dir: Path, model_name: str, filename: str) -> Path | None:
    """Return path to *filename* in the most recent version dir for *model_name*."""
    model_dir = artifacts_dir / model_name
    if not model_dir.exists():
        return None
    versions = sorted(model_dir.iterdir(), reverse=True)
    for v in versions:
        candidate = v / filename
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Meta-feature assembly
# ---------------------------------------------------------------------------

def build_meta_features(
    panel_df: pd.DataFrame,
    artifacts_dir: str = "ml/artifacts",
) -> pd.DataFrame:
    """Assemble the meta-feature matrix from OOF predictions and panel columns.

    Args:
        panel_df:      Full NTA-week panel with Clay PCA columns present.
        artifacts_dir: Root artifact directory.

    Returns:
        DataFrame with columns META_FEATURE_COLS + LABEL_COL, indexed by
        (nta_id, week_start). Rows without all OOF predictions are dropped.
    """
    artifacts = Path(artifacts_dir)
    df = panel_df.copy()
    df["_key"] = df["nta_id"] + "|" + df["week_start"].astype(str)

    # Load CatBoost OOF
    cb_path = _find_latest_artifact(artifacts / "tabular", "catboost", "oof_predictions.json")
    if cb_path is None:
        raise FileNotFoundError(
            "CatBoost OOF predictions not found. Re-run train_tabular.py to generate them."
        )
    cb_oof = _load_oof(cb_path)
    df[CATBOOST_COL] = df["_key"].map(cb_oof)

    # Load TFT OOF
    tft_path = _find_latest_artifact(artifacts / "tft", "tft", "oof_predictions.json")
    if tft_path is None:
        log.warning("TFT OOF not found — filling with CatBoost OOF as proxy.")
        df[TFT_COL] = df[CATBOOST_COL]
    else:
        tft_oof = _load_oof(tft_path)
        df[TFT_COL] = df["_key"].map(tft_oof).fillna(df[CATBOOST_COL])

    # Load Chronos OOF
    ch_path = _find_latest_artifact(artifacts / "chronos", "chronos", "oof_predictions.json")
    if ch_path is None:
        log.warning("Chronos OOF not found — filling with CatBoost OOF as proxy.")
        df[CHRONOS_COL] = df[CATBOOST_COL]
    else:
        ch_oof = _load_oof(ch_path)
        df[CHRONOS_COL] = df["_key"].map(ch_oof).fillna(df[CATBOOST_COL])

    # Clay PCA columns (may not be present if build_clay_embeddings not run yet)
    for col in CLAY_COLS:
        if col not in df.columns:
            df[col] = 0.0

    # Cast Clay cols to float
    for col in CLAY_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Keep only rows with valid OOF predictions
    required = [CATBOOST_COL, TFT_COL, CHRONOS_COL, LABEL_COL]
    df = df.dropna(subset=required)

    # Cast label
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    return df[["nta_id", "week_start"] + META_FEATURE_COLS + [LABEL_COL]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# FusionModel
# ---------------------------------------------------------------------------

@dataclass
class FusionModel:
    """Calibrated stacked meta-learner.

    Wraps an isotonically calibrated sklearn classifier trained on
    meta-features (CatBoost OOF + TFT OOF + Chronos OOF + Clay PCA).

    Attributes:
        estimator:  The underlying fitted (and calibrated) classifier.
        feature_cols: Ordered list of feature column names used at fit time.
        model_type:  ``"logistic_regression"`` or ``"mlp"``.
    """

    estimator: Any
    feature_cols: list[str]
    model_type: str

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated probability array of shape (N, 2)."""
        X_arr = X[self.feature_cols].values.astype("float32")
        return self.estimator.predict_proba(X_arr)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary predictions (threshold 0.5)."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _make_lr_pipeline() -> Any:
    """Logistic regression meta-learner with imputation."""
    from sklearn.impute import SimpleImputer  # noqa: PLC0415
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
    from sklearn.pipeline import Pipeline  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)),
    ])


def _make_mlp_pipeline() -> Any:
    """Shallow MLP meta-learner (ablation)."""
    from sklearn.impute import SimpleImputer  # noqa: PLC0415
    from sklearn.neural_network import MLPClassifier  # noqa: PLC0415
    from sklearn.pipeline import Pipeline  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )),
    ])


def train_fusion(
    meta_df: pd.DataFrame,
    *,
    model_type: str = "logistic_regression",
    holdout_weeks: int = 12,
) -> tuple[FusionModel, dict[str, float]]:
    """Train and calibrate the fusion meta-learner.

    Args:
        meta_df:       Meta-feature DataFrame from :func:`build_meta_features`.
        model_type:    ``"logistic_regression"`` (primary) or ``"mlp"`` (ablation).
        holdout_weeks: Weeks held out for calibration + test evaluation.

    Returns:
        ``(fusion_model, metrics)`` where metrics contains pr_auc, brier, top_decile_lift.
    """
    from sklearn.calibration import CalibratedClassifierCV  # noqa: PLC0415

    from rat_ml.eval.metrics import brier_score, pr_auc, top_decile_lift  # noqa: PLC0415

    # Chronological train/test split
    weeks = sorted(meta_df["week_start"].unique())
    cutoff = weeks[-holdout_weeks] if len(weeks) > holdout_weeks else weeks[0]
    train_df = meta_df[meta_df["week_start"] < cutoff]
    test_df  = meta_df[meta_df["week_start"] >= cutoff]

    X_train = train_df[META_FEATURE_COLS].values.astype("float32")
    y_train = train_df[LABEL_COL].values
    X_test  = test_df[META_FEATURE_COLS].values.astype("float32")
    y_test  = test_df[LABEL_COL].values

    # Base estimator
    if model_type == "logistic_regression":
        base = _make_lr_pipeline()
    elif model_type == "mlp":
        base = _make_mlp_pipeline()
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    # Fit with 5-fold cross-validated isotonic calibration.
    # (cv="prefit" was removed in sklearn>=1.3; cv=5 is the recommended replacement.)
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)

    # Evaluate on test set
    y_prob = calibrated.predict_proba(X_test)[:, 1]
    metrics = {
        "pr_auc": pr_auc(y_test, y_prob),
        "brier": brier_score(y_test, y_prob),
        "top_decile_lift": top_decile_lift(y_test, y_prob),
    }
    log.info(
        "Fusion [%s] — PR-AUC=%.4f  Brier=%.4f  Top-decile lift=%.2fx",
        model_type, metrics["pr_auc"], metrics["brier"], metrics["top_decile_lift"],
    )

    model = FusionModel(
        estimator=calibrated,
        feature_cols=META_FEATURE_COLS,
        model_type=model_type,
    )
    return model, metrics
