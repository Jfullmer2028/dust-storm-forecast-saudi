# Dust-Storm Onset Prediction — Results Report

**Data mode:** real
**Generated:** pipeline run

## Dataset Summary

- Total samples: 3,285
- Positive (dust event next day): 201 (6.1%)
- Stations: hafar, riyadh, sharurah
- Study period: 2018-01-01 00:00:00 to 2020-12-30 00:00:00

## Cross-Validation Results (TimeSeriesSplit, 5 folds)

### Baseline Model (ERA5 + soil + NDVI, no albedo)

| Fold | F2-score |
|------|----------|
| 1 | 0.2852 |
| 2 | 0.2690 |
| 3 | 0.1695 |
| 4 | 0.3770 |
| 5 | 0.2247 |
| **Mean** | **0.2651** |
| Std | 0.0689 |

### Full Model (+ MODIS shortwave albedo anomaly)

| Fold | F2-score |
|------|----------|
| 1 | 0.2792 |
| 2 | 0.2694 |
| 3 | 0.0694 |
| 4 | 0.3770 |
| 5 | 0.2913 |
| **Mean** | **0.2573** |
| Std | 0.1014 |

## Model Comparison

| Metric | Baseline | Full | Delta |
|--------|----------|------|-------|
| Mean F2 (CV) | 0.2651 | 0.2573 | -0.0078 |

## Per-Station Out-of-Fold F2

| Station | n | Positives | Baseline F2 | Full F2 | Delta |
|---------|---|-----------|-------------|---------|-------|
| hafar | 911 | 50 | 0.3187 | 0.3150 | -0.0037 |
| riyadh | 912 | 42 | 0.3053 | 0.2933 | -0.0120 |
| sharurah | 912 | 34 | 0.1579 | 0.2523 | +0.0944 |

## Statistical Tests

### Wilcoxon Signed-Rank Test (per-fold F2 differences)

- Per-fold differences: [-0.006, 0.0004, -0.1, 0.0001, 0.0665]
- W statistic: 7.00
- p-value: 1.0000
- Significant at alpha=0.05: No

### Bootstrap Confidence Interval (5000 resamples)

- Point estimate (median Delta F2): +0.0090
- 95% CI: [-0.0349, +0.0535]
- **Interpretation:** CI straddles 0 — no significant difference detected.

## Figures

- `outputs/f2_comparison_by_fold.png` — per-fold F2 bar chart
- `outputs/bootstrap_delta_f2.png` — bootstrap Delta F2 distribution
- `outputs/shap_importance.png` — SHAP feature importance (full model)

## Conclusion

The full model did not outperform the baseline in this run. With real MODIS/ERA5 data and more dust events, re-evaluate.
