"""Unit tests for the inference pipeline (T-22)."""

from __future__ import annotations

import numpy as np
import pytest

from rat_api.ml.predict import predict_risk


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "complaints_count",
    "complaints_lag_1w",
    "weather_tavg_c",
    "units_total",
]

EVEN_THRESHOLDS = [i / 10 for i in range(1, 11)]  # 0.1 … 1.0


class _FakeModel:
    """Minimal model stub with predict_proba."""

    def __init__(self, proba: float) -> None:
        self._proba = proba

    def predict_proba(self, X):  # noqa: ANN001, N803
        n = len(X)
        return np.array([[1 - self._proba, self._proba]] * n)


def _feature_row(overrides: dict | None = None) -> dict:
    base = {col: 1.0 for col in FEATURE_COLS}
    if overrides:
        base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# predict_risk
# ---------------------------------------------------------------------------


def test_predict_risk_score_range():
    model = _FakeModel(0.72)
    result = predict_risk(model, _feature_row(), FEATURE_COLS, EVEN_THRESHOLDS, "v1")
    assert 0.0 <= result.risk_score <= 1.0


def test_predict_risk_score_value():
    model = _FakeModel(0.72)
    result = predict_risk(model, _feature_row(), FEATURE_COLS, EVEN_THRESHOLDS, "v1")
    assert result.risk_score == pytest.approx(0.72, abs=1e-5)


def test_predict_risk_version_passed_through():
    model = _FakeModel(0.5)
    result = predict_risk(model, _feature_row(), FEATURE_COLS, EVEN_THRESHOLDS, "my-version")
    assert result.model_version == "my-version"


def test_predict_risk_missing_features_filled_with_zero():
    """Feature not in feature_row should be filled with 0, not raise."""
    model = _FakeModel(0.3)
    result = predict_risk(model, {}, FEATURE_COLS, EVEN_THRESHOLDS, "v1")
    assert 0.0 <= result.risk_score <= 1.0


@pytest.mark.parametrize(
    "score, expected_decile",
    [
        # Thresholds are [0.1, 0.2, …, 1.0]; decile = count of thresholds score >= to.
        (0.0, 1),   # below 0.1 → loop never updates → decile stays 1
        (0.05, 1),  # same
        (0.15, 1),  # only >= 0.1 (i=1) → decile=1
        (0.55, 5),  # >= 0.1..0.5 (i=1..5) → decile=5
        (0.99, 9),  # >= 0.1..0.9 (i=1..9), < 1.0 → decile=9
        (1.0, 10),  # >= all 10 thresholds → decile=10
    ],
)
def test_predict_risk_decile_assignment(score: float, expected_decile: int):
    model = _FakeModel(score)
    result = predict_risk(model, _feature_row(), FEATURE_COLS, EVEN_THRESHOLDS, "v1")
    assert result.risk_decile == expected_decile


def test_predict_risk_decile_never_exceeds_10():
    model = _FakeModel(1.0)
    result = predict_risk(model, _feature_row(), FEATURE_COLS, EVEN_THRESHOLDS, "v1")
    assert result.risk_decile <= 10


def test_predict_risk_top_factors_list():
    """SHAP will fail on a fake model — top_factors should be empty, not raise."""
    model = _FakeModel(0.4)
    result = predict_risk(model, _feature_row(), FEATURE_COLS, EVEN_THRESHOLDS, "v1")
    assert isinstance(result.top_factors, list)
