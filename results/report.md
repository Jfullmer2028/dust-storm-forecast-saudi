# Dust-Storm Onset Prediction — Results Report

**Data mode:** synthetic
**Generated:** pipeline run

## Dataset Summary

- Total samples: 14,576
- Positive (dust event next day): 1,239 (8.5%)
- Stations: arar, dammam, hafar, najran, qassim, riyadh, sharurah, tabuk
- Study period: 2018-01-01 00:00:00 to 2022-12-27 00:00:00

## Cross-Validation Results (TimeSeriesSplit, 8 folds)

### Baseline Model (ERA5 + soil + NDVI, no albedo)

| Fold | F2-score |
|------|----------|
| 1 | 0.4627 |
| 2 | 0.4815 |
| 3 | 0.4599 |
| 4 | 0.4662 |
| 5 | 0.3795 |
| 6 | 0.5585 |
| 7 | 0.4510 |
| 8 | 0.5517 |
| **Mean** | **0.4764** |
| Std | 0.0537 |

### Full Model (+ MODIS albedo anomaly within 200 km)

| Fold | F2-score |
|------|----------|
| 1 | 0.4898 |
| 2 | 0.5269 |
| 3 | 0.5919 |
| 4 | 0.5776 |
| 5 | 0.5077 |
| 6 | 0.6276 |
| 7 | 0.5635 |
| 8 | 0.5861 |
| **Mean** | **0.5589** |
| Std | 0.0438 |

## Model Comparison

| Metric | Baseline | Full | Delta |
|--------|----------|------|-------|
| Mean F2 (CV) | 0.4764 | 0.5589 | +0.0825 |

## Per-Station Out-of-Fold F2

| Station | n | Positives | Baseline F2 | Full F2 | Delta |
|---------|---|-----------|-------------|---------|-------|
| arar | 1619 | 115 | 0.4089 | 0.5153 | +0.1064 |
| dammam | 1619 | 127 | 0.4691 | 0.5361 | +0.0670 |
| hafar | 1619 | 173 | 0.5230 | 0.6018 | +0.0788 |
| najran | 1619 | 192 | 0.5472 | 0.6270 | +0.0798 |
| qassim | 1619 | 124 | 0.4769 | 0.5347 | +0.0578 |
| riyadh | 1619 | 132 | 0.4611 | 0.5805 | +0.1193 |
| sharurah | 1619 | 107 | 0.4771 | 0.5072 | +0.0301 |
| tabuk | 1619 | 132 | 0.4431 | 0.5319 | +0.0888 |

## Statistical Tests

### Wilcoxon Signed-Rank Test (per-fold F2 differences)

- Per-fold differences: [0.0271, 0.0454, 0.132, 0.1113, 0.1283, 0.0691, 0.1124, 0.0344]
- W statistic: 0.00
- p-value: 0.0078
- Significant at alpha=0.05: Yes

### Bootstrap Confidence Interval (5000 resamples)

- Point estimate (median Delta F2): +0.0794
- 95% CI: [+0.0623, +0.0965]
- **Interpretation:** CI entirely above 0 — full model significantly better.

## Figures

- `outputs/f2_comparison_by_fold.png` — per-fold F2 bar chart
- `outputs/bootstrap_delta_f2.png` — bootstrap Delta F2 distribution
- `outputs/shap_importance.png` — SHAP feature importance (full model)

## Conclusion

Adding dynamic MODIS shortwave broadband albedo anomaly features (within 200 km of each station) improves 24-hour dust-storm onset prediction F2-score relative to the baseline meteorological model. Both the mean CV improvement and bootstrap 95% CI support this finding.
