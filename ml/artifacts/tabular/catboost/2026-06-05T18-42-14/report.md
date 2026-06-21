# catboost — Training Report

## CV Results

| Fold | PR-AUC | ROC-AUC | Brier | Top-Decile Lift |
|---|---:|---:|---:|---:|
| 1 | 0.7989 | 0.7675 | 0.1985 | 1.63 |
| 2 | 0.8258 | 0.7896 | 0.1924 | 1.83 |
| 3 | 0.8157 | 0.7787 | 0.1932 | 1.74 |
| 4 | 0.8325 | 0.7810 | 0.1872 | 1.57 |
| 5 | 0.7627 | 0.7437 | 0.2013 | 1.47 |

**CV PR-AUC**: 0.8071 ± 0.0249

## Test Set Results

| Metric | Value |
|---|---:|
| PR-AUC | 0.7947 |
| ROC-AUC | 0.7556 |
| Brier Score | 0.2070 |
| Top-Decile Lift | 1.53 |

## Top SHAP Features

| Feature | Mean |SHAP| |
|---|---:|
| `year_built_median` | 0.390256 |
| `borough` | 0.386923 |
| `units_total` | 0.333977 |
| `complaints_count` | 0.169063 |
| `neighbor_active_rat_signs_rate_lag_1w` | 0.165633 |
| `complaints_lag_1w` | 0.139232 |
| `complaints_lag_4w` | 0.135251 |
| `neighbor_complaints_count_lag_4w` | 0.115961 |
| `weather_hdd` | 0.068418 |
| `permits_active_count` | 0.065809 |
| `complaints_lag_12w` | 0.053315 |
| `weather_cdd` | 0.049527 |
| `weather_prcp_mm` | 0.044216 |
| `weather_tavg_c` | 0.034354 |
| `regime_commercial_containerization` | 0.026172 |
| `regime_residential_containerization` | 0.020466 |
| `rest_pest_violations_count` | 0.014658 |
| `demolitions_count` | 0.000000 |
| `landuse_residential_pct` | 0.000000 |
| `landuse_commercial_pct` | 0.000000 |
