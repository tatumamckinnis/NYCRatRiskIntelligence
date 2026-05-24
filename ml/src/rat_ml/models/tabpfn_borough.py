"""Per-borough TabPFN v2 models for small boroughs (T-17).

For boroughs with ≤ 10,000 training rows, a TabPFNClassifier is trained
on the borough subset.  At inference time, each row is routed to its
borough model when available; rows from larger boroughs fall back to the
main CatBoost model.

Usage::

    from rat_ml.models.tabpfn_borough import BoroughTabPFNEnsemble

    ensemble = BoroughTabPFNEnsemble(fallback_model=catboost_result.model)
    ensemble.fit(train_df, feature_cols)
    probs = ensemble.predict_proba(test_df[feature_cols])
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rat_ml.eval.metrics import metric_bundle
from rat_ml.models.tabular import IsotonicCalibratedClassifier

MAX_ROWS_FOR_TABPFN = 10_000
BOROUGH_COL = "borough"
LABEL_COL = "active_rat_signs_ind"


class BoroughTabPFNEnsemble:
    """Routes inference to per-borough TabPFN models for small boroughs.

    Args:
        fallback_model: A fitted model with ``predict_proba(X)`` used for
                        boroughs that exceed *max_rows* or lack a TabPFN fit.
        max_rows:       Borough row threshold below which TabPFN is trained.
    """

    def __init__(
        self,
        fallback_model: Any,
        max_rows: int = MAX_ROWS_FOR_TABPFN,
    ) -> None:
        self.fallback_model = fallback_model
        self.max_rows = max_rows
        self._borough_models: dict[str, IsotonicCalibratedClassifier] = {}
        self._feature_cols: list[str] = []
        self.fit_report: dict[str, dict] = {}

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str = LABEL_COL,
    ) -> "BoroughTabPFNEnsemble":
        """Train TabPFN on qualifying boroughs.

        Boroughs with > *max_rows* training rows are skipped (fallback used).
        TabPFN v2 caps its training set internally at 10k rows; we apply the
        same cap here to keep training fast.

        Args:
            train_df:     Training DataFrame with a ``borough`` column.
            feature_cols: Feature column names.
            label_col:    Binary label column name.
        """
        try:
            from tabpfn import TabPFNClassifier  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "tabpfn is required for BoroughTabPFNEnsemble. "
                "Install with: uv sync --extra ml"
            ) from e

        self._feature_cols = feature_cols
        non_cat_cols = [c for c in feature_cols if c != BOROUGH_COL]

        for borough, group in train_df.groupby(BOROUGH_COL):
            n = len(group)
            if n > self.max_rows:
                self.fit_report[str(borough)] = {
                    "status": "skipped_too_large",
                    "n_rows": n,
                }
                continue

            X = group[non_cat_cols].fillna(0).astype(float)
            y = group[label_col].astype(int)

            if y.nunique() < 2:
                self.fit_report[str(borough)] = {
                    "status": "skipped_single_class",
                    "n_rows": n,
                }
                continue

            base = TabPFNClassifier(device="cpu", n_estimators=8)
            base.fit(X, y)

            # Use a 20% hold-out from the borough for calibration
            cal_n = max(10, n // 5)
            cal_idx = group.index[-cal_n:]
            X_cal = group.loc[cal_idx, non_cat_cols].fillna(0).astype(float)
            y_cal = group.loc[cal_idx, label_col].astype(int)

            calibrated = IsotonicCalibratedClassifier(base)
            calibrated.fit(X_cal, y_cal)
            self._borough_models[str(borough)] = calibrated

            raw_probs = calibrated.predict_proba(X)[:, 1]
            self.fit_report[str(borough)] = {
                "status": "trained",
                "n_rows": n,
                "metrics": metric_bundle(y.values, raw_probs),
            }

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated probabilities, routing by borough.

        Args:
            X: Feature DataFrame; must contain ``borough`` and all
               ``feature_cols`` passed to :meth:`fit`.

        Returns:
            Array of shape ``(n_samples, 2)`` with [P(0), P(1)] columns.
        """
        non_cat_cols = [c for c in self._feature_cols if c != BOROUGH_COL]
        probs = np.zeros(len(X))

        for borough, group_idx in X.groupby(BOROUGH_COL).groups.items():
            rows = X.loc[group_idx]
            if str(borough) in self._borough_models:
                p = self._borough_models[str(borough)].predict_proba(
                    rows[non_cat_cols].fillna(0).astype(float)
                )[:, 1]
            else:
                p = self.fallback_model.predict_proba(rows[self._feature_cols])[:, 1]
            probs[X.index.get_indexer(group_idx)] = p

        return np.column_stack([1.0 - probs, probs])
