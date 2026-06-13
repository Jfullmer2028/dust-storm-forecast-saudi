# Dust-Storm Onset Prediction — Results Report

**Data mode:** real
**Generated:** pipeline run

## Dataset Summary

- Total samples: 3,285
- Positive (dust event next day): 201 (6.1%)
- Stations: hafar, riyadh, sharurah
- Study period: 2018-01-01 00:00:00 to 2020-12-30 00:00:00

## Headline Comparison

Primary metric is **PR-AUC (average precision)** — threshold-independent and appropriate for this rare-event problem. ROC-AUC is reported alongside it; F2 at the per-fold tuned threshold is the **operational** metric. All deltas are full − baseline with paired bootstrap 95% CIs (out-of-fold predictions, 5-fold TimeSeriesSplit).

| Metric | Baseline | Full | Δ | 95% CI | Verdict |
|--------|----------|------|---|--------|---------|
| **PR-AUC** (primary) | 0.1170 | 0.1155 | -0.0015 | [-0.0210, +0.0184] | CI straddles 0 — no significant difference |
| ROC-AUC | 0.7133 | 0.7179 | +0.0045 | [-0.0150, +0.0248] | CI straddles 0 — no significant difference |
| F2 @ tuned thr (operational) | 0.2651 | 0.2573 | -0.0078 | [-0.0349, +0.0535] | CI straddles 0 — no significant difference |

## Per-Fold Cross-Validation (TimeSeriesSplit, 5 folds)

| Fold | Baseline PR-AUC | Full PR-AUC | Baseline F2 | Full F2 |
|------|-----------------|-------------|-------------|---------|
| 1 | 0.2367 | 0.2188 | 0.2852 | 0.2792 |
| 2 | 0.0949 | 0.0908 | 0.2690 | 0.2694 |
| 3 | 0.0599 | 0.0759 | 0.1695 | 0.0694 |
| 4 | 0.1620 | 0.1558 | 0.3770 | 0.3770 |
| 5 | 0.0949 | 0.0885 | 0.2247 | 0.2913 |
| **Mean** | **0.1297** | **0.1260** | **0.2651** | **0.2573** |

## Per-Station Out-of-Fold Performance

| Station | n | Positives | Base PR-AUC | Full PR-AUC | ΔPR-AUC | Base F2 | Full F2 |
|---------|---|-----------|-------------|-------------|---------|---------|---------|
| hafar | 911 | 50 | 0.1542 | 0.1534 | -0.0008 | 0.3187 | 0.3150 |
| riyadh | 912 | 42 | 0.1205 | 0.1155 | -0.0050 | 0.3053 | 0.2933 |
| sharurah | 912 | 34 | 0.0889 | 0.0948 | +0.0060 | 0.1579 | 0.2523 |

## Statistical Tests

- **Paired bootstrap (5 000 resamples)** on out-of-fold predictions gives the 95% CIs in the headline table — the primary inference.
- **Wilcoxon signed-rank on per-fold PR-AUC** (n=5): W=4.00, p=0.4375 (not significant at α=0.05).
- **Wilcoxon signed-rank on per-fold F2** (n=5): W=7.00, p=1.0000 (not significant at α=0.05).

## Figures

- `pr_curves.png` — precision–recall curves (baseline vs full)
- `f2_comparison_by_fold.png` — per-fold F2 bar chart
- `bootstrap_delta_f2.png` — bootstrap ΔF2 distribution
- `shap_importance.png` — SHAP feature importance (full model)

## Conclusion

On real observations the MODIS albedo anomaly does not improve PR-AUC (-0.0015, 95% CI [-0.0210, +0.0184]). Satellite albedo provides no significant incremental skill over the meteorological baseline at these stations — an honest null result, with station-level heterogeneity worth follow-up.
