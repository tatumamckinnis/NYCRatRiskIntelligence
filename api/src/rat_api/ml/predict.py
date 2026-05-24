"""Inference pipeline: feature row → risk score + SHAP top factors (T-19)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rat_api.models.risk import RiskFactor

# Human-readable labels for feature columns shown in the API response.
FEATURE_LABELS: dict[str, str] = {
    "complaints_count": "311 rodent complaints this week",
    "complaints_lag_1w": "311 complaints 1 week ago",
    "complaints_lag_4w": "311 complaints 4 weeks ago",
    "complaints_lag_12w": "311 complaints 12 weeks ago",
    "rest_pest_violations_count": "Restaurant pest violations this week",
    "permits_active_count": "Active construction permits",
    "demolitions_count": "Demolition permits this week",
    "weather_tavg_c": "Average temperature (°C)",
    "weather_prcp_mm": "Weekly precipitation (mm)",
    "weather_hdd": "Heating degree days",
    "weather_cdd": "Cooling degree days",
    "units_total": "Total residential + commercial units",
    "year_built_median": "Median building year",
    "landuse_residential_pct": "Residential land use fraction",
    "landuse_commercial_pct": "Commercial land use fraction",
    "neighbor_active_rat_signs_rate_lag_1w": "Neighboring NTA infestation rate (1 wk ago)",
    "neighbor_complaints_count_lag_4w": "Neighboring NTA complaints (4 wks ago)",
    "regime_covid": "COVID-era regime",
    "regime_8pm_setout": "8 PM set-out policy active",
    "regime_commercial_containerization": "Commercial containerization active",
    "regime_residential_containerization": "Residential containerization active",
    "regime_rmz_active": "Rat Mitigation Zone active",
    "borough": "Borough",
}


@dataclass
class PredictionResult:
    risk_score: float
    risk_decile: int
    top_factors: list[RiskFactor]
    model_version: str


def _compute_shap_factors(
    model: object,
    X: pd.DataFrame,
    feature_cols: list[str],
    n: int = 5,
) -> list[RiskFactor]:
    """Compute SHAP values for one row and return the top-n factors."""
    try:
        import shap  # noqa: PLC0415

        # Unwrap IsotonicCalibratedClassifier to get the base estimator
        base = getattr(model, "base_estimator", model)
        explainer = shap.TreeExplainer(base)
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        row_shap = shap_vals[0]
    except Exception:  # noqa: BLE001
        # SHAP unavailable or failed — return empty factors gracefully
        return []

    order = np.argsort(np.abs(row_shap))[::-1][:n]
    factors: list[RiskFactor] = []
    for idx in order:
        col = feature_cols[idx]
        val = float(row_shap[idx])
        factors.append(
            RiskFactor(
                feature=col,
                contribution=round(val, 6),
                direction="up" if val > 0 else "down",
                readable=FEATURE_LABELS.get(col, col),
            )
        )
    return factors


def predict_risk(
    model: object,
    feature_row: dict,
    feature_cols: list[str],
    decile_thresholds: list[float],
    model_version: str,
) -> PredictionResult:
    """Run inference on a single feature row.

    Args:
        model:              Fitted model with ``predict_proba(X)`` method.
        feature_row:        Dict from :func:`~rat_api.ml.features.get_nta_features`.
        feature_cols:       Ordered list of feature column names the model expects.
        decile_thresholds:  10 score thresholds (from weekly score distribution)
                            used to compute ``risk_decile``.
        model_version:      Version string from the model metadata.

    Returns:
        :class:`PredictionResult`
    """
    X = pd.DataFrame([{col: feature_row.get(col) for col in feature_cols}])
    # Fill missing values with 0 so the model can always run
    X = X.fillna(0)

    probs = model.predict_proba(X)  # type: ignore[union-attr]
    risk_score = float(probs[0, 1])

    # Decile: find which bucket the score falls into
    decile = 1
    for i, threshold in enumerate(sorted(decile_thresholds), start=1):
        if risk_score >= threshold:
            decile = i
    risk_decile = min(decile, 10)

    top_factors = _compute_shap_factors(model, X, feature_cols)

    return PredictionResult(
        risk_score=round(risk_score, 6),
        risk_decile=risk_decile,
        top_factors=top_factors,
        model_version=model_version,
    )
