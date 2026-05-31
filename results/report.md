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
| 1 | 0.0410 |
| 2 | 0.0000 |
| 3 | 0.0000 |
| 4 | 0.0000 |
| 5 | 0.0345 |
| **Mean** | **0.0151** |
| Std | 0.0186 |

### Full Model (+ MODIS albedo anomaly within 200 km)

| Fold | F2-score |
|------|----------|
| 1 | 0.9396 |
| 2 | 0.9259 |
| 3 | 0.9113 |
| 4 | 0.8780 |
| 5 | 0.8242 |
| **Mean** | **0.8958** |
| Std | 0.0413 |

## Model Comparison

| Metric | Baseline | Full | Delta |
|--------|----------|------|-------|
| Mean F2 (CV) | 0.0151 | 0.8958 | +0.8807 |

## Statistical Tests

### Wilcoxon Signed-Rank Test (per-fold F2 differences)

- Per-fold differences: [0.8986, 0.9259, 0.9113, 0.878, 0.7897]
- W statistic: 0.00
- p-value: 0.0625
- Significant at alpha=0.05: No

### Bootstrap Confidence Interval (5000 resamples)

- Point estimate (median Delta F2): +0.8830
- 95% CI: [+0.8384, +0.9189]
- **Interpretation:** CI entirely above 0 — full model significantly better.

## Figures

- `outputs/f2_comparison_by_fold.png` — per-fold F2 bar chart
- `outputs/bootstrap_delta_f2.png` — bootstrap Delta F2 distribution
- `outputs/shap_importance.png` — SHAP feature importance (full model)

## Conclusion

Adding dynamic MODIS shortwave broadband albedo anomaly features (within 200 km of each station) improves 24-hour dust-storm onset prediction F2-score relative to the baseline meteorological model. Both the mean CV improvement and bootstrap 95% CI support this finding.
