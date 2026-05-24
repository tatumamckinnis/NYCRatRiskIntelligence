"""Evaluation metrics for the tabular risk model (T-14).

All functions accept plain numpy arrays or array-likes and return
plain Python floats / dicts so results can be serialised directly
to the artifact report.md.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area under the Precision-Recall curve (average precision).

    Preferred over ROC-AUC for imbalanced binary labels.
    """
    return float(average_precision_score(y_true, y_prob))


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area under the ROC curve."""
    return float(roc_auc_score(y_true, y_prob))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score (mean squared error of probability forecasts).

    Lower is better; a perfect model scores 0.0; predicting the base
    rate scores ``base_rate * (1 - base_rate)``.
    """
    return float(brier_score_loss(y_true, y_prob))


def top_decile_lift(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Fraction of positive labels captured in the top-scoring 10% of rows.

    Lift = (positives in top decile / total positives) / 0.10

    A random model scores ~1.0; the target is ≥ 4.0 (≥ 40% of positives
    in the top 10% of predictions, per SPEC §15).
    """
    n = len(y_true)
    if n == 0:
        return 0.0
    top_n = max(1, n // 10)
    order = np.argsort(y_prob)[::-1]
    top_positives = int(np.sum(y_true[order[:top_n]]))
    total_positives = int(np.sum(y_true))
    if total_positives == 0:
        return 0.0
    return float(top_positives / total_positives / 0.10)


def calibration_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float | list[float]]:
    """Reliability diagram data + expected calibration error.

    Returns a dict suitable for JSON serialisation:
    ``{ece, fraction_of_positives, mean_predicted_value}``.
    """
    fraction_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    ece = float(np.mean(np.abs(fraction_pos - mean_pred)))
    return {
        "ece": ece,
        "fraction_of_positives": fraction_pos.tolist(),
        "mean_predicted_value": mean_pred.tolist(),
    }


def metric_bundle(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Compute the full metric set for one model/fold.

    Returns:
        ``{pr_auc, roc_auc, brier, top_decile_lift}``
    """
    return {
        "pr_auc": pr_auc(y_true, y_prob),
        "roc_auc": roc_auc(y_true, y_prob),
        "brier": brier_score(y_true, y_prob),
        "top_decile_lift": top_decile_lift(y_true, y_prob),
    }
