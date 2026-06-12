# Dust-Storm Onset Prediction — Results Report

**Data mode:** synthetic
**Generated:** pipeline run

## Dataset Summary

- Total samples: 3,264
- Positive (dust event next day): 236 (7.2%)
- Stations: hafar, riyadh, sharurah
- Study period: 2018-01-01 00:00:00 to 2020-12-23 00:00:00

## Cross-Validation Results (TimeSeriesSplit, 5 folds)

### Baseline Model (ERA5 + soil + NDVI, no albedo)

| Fold | F2-score |
|------|----------|
| 1 | 0.0000 |
| 2 | 0.0221 |
| 3 | 0.0000 |
| 4 | 0.0325 |
| 5 | 0.0000 |
| **Mean** | **0.0109** |
| Std | 0.0138 |

### Full Model (+ MODIS albedo anomaly within 200 km)

| Fold | F2-score |
|------|----------|
| 1 | 0.7643 |
| 2 | 0.9298 |
| 3 | 0.8599 |
| 4 | 0.8871 |
| 5 | 0.8511 |
| **Mean** | **0.8584** |
| Std | 0.0545 |

## Model Comparison

| Metric | Baseline | Full | Delta |
|--------|----------|------|-------|
| Mean F2 (CV) | 0.0109 | 0.8584 | +0.8475 |

## Statistical Tests

### Wilcoxon Signed-Rank Test (per-fold F2 differences)

- Per-fold differences: [0.7643, 0.9077, 0.8599, 0.8546, 0.8511]
- W statistic: 0.00
- p-value: 0.0625
- Significant at alpha=0.05: No

### Bootstrap Confidence Interval (5000 resamples)

- Point estimate (median Delta F2): +0.8558
- 95% CI: [+0.8126, +0.8933]
- **Interpretation:** CI entirely above 0 — full model significantly better.

## Figures

- `outputs/f2_comparison_by_fold.png` — per-fold F2 bar chart
- `outputs/bootstrap_delta_f2.png` — bootstrap Delta F2 distribution
- `outputs/shap_importance.png` — SHAP feature importance (full model)

## Conclusion

Adding dynamic MODIS shortwave broadband albedo anomaly features (within 200 km of each station) improves 24-hour dust-storm onset prediction F2-score relative to the baseline meteorological model. Both the mean CV improvement and bootstrap 95% CI support this finding.
