"""Tabular model trainers: CatBoost, LightGBM, Logistic Regression (T-16).

Each trainer follows the same interface:
1. Expanding-window TS-CV (5 folds, 28-day gap) → fold-by-fold metrics
2. Retrain on the full training set
3. Isotonic calibration on the final fold's validation set
4. SHAP values for the top-20 most important features
5. Final evaluation on the held-out test set

Usage::

    from rat_ml.models.tabular import CatBoostTrainer, LightGBMTrainer, LRTrainer

    trainer = CatBoostTrainer()
    result = trainer.fit(train_df, test_df, feature_cols, label_col)
    print(result.test_metrics)
    print(result.top_shap_features)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from rat_ml.eval.metrics import metric_bundle
from rat_ml.eval.timeseries_cv import expanding_window_splits


class IsotonicCalibratedClassifier:
    """Wraps a pre-fitted classifier with an isotonic calibration layer.

    Replaces ``CalibratedClassifierCV(cv="prefit")`` which was removed in
    scikit-learn 1.4.  Fits an ``IsotonicRegression`` on the base model's
    raw predicted probabilities, then applies it at predict time.
    """

    def __init__(self, base_estimator: Any) -> None:
        self.base_estimator = base_estimator
        self._iso: IsotonicRegression | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "IsotonicCalibratedClassifier":
        raw = self.base_estimator.predict_proba(X)[:, 1]
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._iso.fit(raw, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._iso is None:
            raise RuntimeError("Call fit() before predict_proba().")
        raw = self.base_estimator.predict_proba(X)[:, 1]
        cal = self._iso.transform(raw)
        return np.column_stack([1.0 - cal, cal])


@dataclass
class TrainResult:
    """Output of a :meth:`BaseTabularTrainer.fit` call."""

    model_name: str
    model: Any                                      # calibrated estimator
    label_encoder: LabelEncoder | None              # borough encoder (LR only)
    fold_metrics: list[dict[str, float]]            # one dict per CV fold
    cv_pr_auc_mean: float
    cv_pr_auc_std: float
    test_metrics: dict[str, float]                  # final test-set metrics
    top_shap_features: dict[str, float]             # {feature: mean_abs_shap}
    feature_cols: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shap_top_n(
    model_obj: Any,
    X: pd.DataFrame,
    feature_cols: list[str],
    n: int = 20,
) -> dict[str, float]:
    """Compute mean |SHAP| values and return the top-n features."""
    try:
        import shap  # noqa: PLC0415
    except ImportError:
        # SHAP not installed — return empty dict; callers handle gracefully.
        return {}

    try:
        explainer = shap.TreeExplainer(model_obj)
        shap_values = explainer.shap_values(X)
        # For binary classification some libs return list[array]; take the
        # positive-class array.
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        mean_abs = np.abs(shap_values).mean(axis=0)
    except Exception:  # noqa: BLE001
        try:
            explainer = shap.LinearExplainer(model_obj, X)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            mean_abs = np.abs(shap_values).mean(axis=0)
        except Exception:  # noqa: BLE001
            return {}

    order = np.argsort(mean_abs)[::-1][:n]
    return {feature_cols[i]: float(mean_abs[i]) for i in order}


def _encode_categoricals(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, LabelEncoder]]:
    """Label-encode categorical columns for sklearn estimators."""
    encoders: dict[str, LabelEncoder] = {}
    X_tr = X_train.copy()
    X_v = X_val.copy()
    for col in cat_cols:
        le = LabelEncoder()
        X_tr[col] = le.fit_transform(X_tr[col].astype(str))
        X_v[col] = X_v[col].astype(str).map(
            lambda x, le=le: le.transform([x])[0]  # noqa: B023
            if x in set(le.classes_)
            else -1
        )
        encoders[col] = le
    return X_tr, X_v, encoders


# ---------------------------------------------------------------------------
# Base trainer
# ---------------------------------------------------------------------------

class BaseTabularTrainer:
    model_name: str = "base"
    _cat_cols: list[str] = []

    def _make_estimator(self) -> Any:
        raise NotImplementedError

    def _prepare_X(
        self,
        X: pd.DataFrame,
        encoders: dict[str, LabelEncoder] | None = None,
    ) -> pd.DataFrame:
        """Apply any pre-processing needed for this estimator type."""
        return X

    def fit(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str = "active_rat_signs_ind",
        n_folds: int = 5,
        gap_days: int = 28,
    ) -> TrainResult:
        """Run CV, retrain, calibrate, compute SHAP, evaluate on test set."""
        from rat_ml.features.feature_matrix import CAT_FEATURE_COLS  # noqa: PLC0415

        cat_cols = [c for c in CAT_FEATURE_COLS if c in feature_cols]

        # ------------------------------------------------------------------
        # Expanding-window CV
        # ------------------------------------------------------------------
        fold_metrics: list[dict[str, float]] = []
        last_val_idx: pd.Index | None = None

        for split in expanding_window_splits(
            train_df, n_folds=n_folds, gap_days=gap_days
        ):
            X_tr = train_df.loc[split.train_idx, feature_cols].copy()
            y_tr = train_df.loc[split.train_idx, label_col].astype(int)
            X_v = train_df.loc[split.val_idx, feature_cols].copy()
            y_v = train_df.loc[split.val_idx, label_col].astype(int)

            X_tr, X_v, _ = _encode_categoricals(X_tr, X_v, cat_cols)

            est = self._make_estimator()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(X_tr, y_tr)

            y_prob = est.predict_proba(X_v)[:, 1]
            fold_metrics.append(metric_bundle(y_v.values, y_prob))
            last_val_idx = split.val_idx

        cv_pr_aucs = [m["pr_auc"] for m in fold_metrics]

        # ------------------------------------------------------------------
        # Retrain on full training set
        # ------------------------------------------------------------------
        X_full = train_df[feature_cols].copy()
        y_full = train_df[label_col].astype(int)

        # Use the last fold's val set as calibration set
        if last_val_idx is not None:
            X_cal = train_df.loc[last_val_idx, feature_cols].copy()
            y_cal = train_df.loc[last_val_idx, label_col].astype(int)
        else:
            X_cal = X_full.copy()
            y_cal = y_full.copy()

        X_full_enc, X_cal_enc, encoders = _encode_categoricals(
            X_full, X_cal, cat_cols
        )

        base_est = self._make_estimator()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base_est.fit(X_full_enc, y_full)

        # Isotonic calibration
        calibrated = IsotonicCalibratedClassifier(base_est)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calibrated.fit(X_cal_enc, y_cal)

        # ------------------------------------------------------------------
        # SHAP (on a sample to keep it fast)
        # ------------------------------------------------------------------
        sample_size = min(500, len(X_full_enc))
        X_shap = X_full_enc.sample(sample_size, random_state=42)
        top_shap = _shap_top_n(base_est, X_shap, feature_cols)

        # ------------------------------------------------------------------
        # Test set evaluation
        # ------------------------------------------------------------------
        X_test = test_df[feature_cols].copy()
        y_test = test_df[label_col].astype(int)
        X_test_enc, _, _ = _encode_categoricals(X_test, X_test.iloc[:1].copy(), cat_cols)

        y_prob_test = calibrated.predict_proba(X_test_enc)[:, 1]
        test_metrics = metric_bundle(y_test.values, y_prob_test)

        le = encoders.get("borough") if encoders else None

        return TrainResult(
            model_name=self.model_name,
            model=calibrated,
            label_encoder=le,
            fold_metrics=fold_metrics,
            cv_pr_auc_mean=float(np.mean(cv_pr_aucs)),
            cv_pr_auc_std=float(np.std(cv_pr_aucs)),
            test_metrics=test_metrics,
            top_shap_features=top_shap,
            feature_cols=feature_cols,
        )


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------

class CatBoostTrainer(BaseTabularTrainer):
    model_name = "catboost"

    def __init__(self, iterations: int = 500, learning_rate: float = 0.05) -> None:
        self.iterations = iterations
        self.learning_rate = learning_rate

    def _make_estimator(self) -> Any:
        from catboost import CatBoostClassifier  # noqa: PLC0415

        from rat_ml.features.feature_matrix import CAT_FEATURE_COLS  # noqa: PLC0415

        return CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            loss_function="Logloss",
            eval_metric="AUC",
            cat_features=CAT_FEATURE_COLS,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )

    def _prepare_X(
        self,
        X: pd.DataFrame,
        encoders: dict[str, LabelEncoder] | None = None,
    ) -> pd.DataFrame:
        # CatBoost handles cat columns natively — no encoding needed
        return X

    def fit(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str = "active_rat_signs_ind",
        n_folds: int = 5,
        gap_days: int = 28,
    ) -> TrainResult:
        """CatBoost-specific fit: passes cat_features, no label encoding."""
        from catboost import CatBoostClassifier  # noqa: PLC0415
        from rat_ml.features.feature_matrix import CAT_FEATURE_COLS  # noqa: PLC0415

        cat_cols = [c for c in CAT_FEATURE_COLS if c in feature_cols]
        fold_metrics: list[dict[str, float]] = []
        last_val_idx: pd.Index | None = None

        for split in expanding_window_splits(
            train_df, n_folds=n_folds, gap_days=gap_days
        ):
            X_tr = train_df.loc[split.train_idx, feature_cols].copy()
            y_tr = train_df.loc[split.train_idx, label_col].astype(int)
            X_v = train_df.loc[split.val_idx, feature_cols].copy()
            y_v = train_df.loc[split.val_idx, label_col].astype(int)

            est = CatBoostClassifier(
                iterations=self.iterations,
                learning_rate=self.learning_rate,
                loss_function="Logloss",
                eval_metric="AUC",
                cat_features=cat_cols,
                random_seed=42,
                verbose=False,
                allow_writing_files=False,
            )
            est.fit(X_tr, y_tr, eval_set=(X_v, y_v))
            y_prob = est.predict_proba(X_v)[:, 1]
            fold_metrics.append(metric_bundle(y_v.values, y_prob))
            last_val_idx = split.val_idx

        cv_pr_aucs = [m["pr_auc"] for m in fold_metrics]

        X_full = train_df[feature_cols].copy()
        y_full = train_df[label_col].astype(int)
        X_cal = (
            train_df.loc[last_val_idx, feature_cols].copy()
            if last_val_idx is not None
            else X_full.copy()
        )
        y_cal = (
            train_df.loc[last_val_idx, label_col].astype(int)
            if last_val_idx is not None
            else y_full.copy()
        )

        base_est = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            loss_function="Logloss",
            cat_features=cat_cols,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
        base_est.fit(X_full, y_full)

        calibrated = IsotonicCalibratedClassifier(base_est)
        calibrated.fit(X_cal, y_cal)

        sample_size = min(500, len(X_full))
        X_shap = X_full.sample(sample_size, random_state=42)
        top_shap = _shap_top_n(base_est, X_shap, feature_cols)

        X_test = test_df[feature_cols].copy()
        y_test = test_df[label_col].astype(int)
        y_prob_test = calibrated.predict_proba(X_test)[:, 1]
        test_metrics = metric_bundle(y_test.values, y_prob_test)

        return TrainResult(
            model_name=self.model_name,
            model=calibrated,
            label_encoder=None,
            fold_metrics=fold_metrics,
            cv_pr_auc_mean=float(np.mean(cv_pr_aucs)),
            cv_pr_auc_std=float(np.std(cv_pr_aucs)),
            test_metrics=test_metrics,
            top_shap_features=top_shap,
            feature_cols=feature_cols,
        )


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

class LightGBMTrainer(BaseTabularTrainer):
    model_name = "lightgbm"

    def __init__(self, n_estimators: int = 500, learning_rate: float = 0.05) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def _make_estimator(self) -> Any:
        from lightgbm import LGBMClassifier  # noqa: PLC0415

        return LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            objective="binary",
            metric="average_precision",
            random_state=42,
            verbose=-1,
        )


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

class LRTrainer(BaseTabularTrainer):
    model_name = "logistic_regression"

    def __init__(self, C: float = 1.0) -> None:
        self.C = C

    def _make_estimator(self) -> Any:
        return LogisticRegression(
            C=self.C,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )
