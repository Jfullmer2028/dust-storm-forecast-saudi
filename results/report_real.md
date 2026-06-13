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
| **PR-AUC** (primary) | 0.1166 | 0.1206 | +0.0040 | [-0.0147, +0.0230] | CI straddles 0 — no significant difference |
| ROC-AUC | 0.7267 | 0.7416 | +0.0149 | [-0.0027, +0.0318] | CI straddles 0 — no significant difference |
| F2 @ tuned thr (operational) | 0.2590 | 0.2335 | -0.0255 | [-0.0709, +0.0196] | CI straddles 0 — no significant difference |

## Per-Fold Cross-Validation (TimeSeriesSplit, 5 folds)

| Fold | Baseline PR-AUC | Full PR-AUC | Baseline F2 | Full F2 |
|------|-----------------|-------------|-------------|---------|
| 1 | 0.1613 | 0.2492 | 0.2510 | 0.2834 |
| 2 | 0.1164 | 0.1053 | 0.2690 | 0.2281 |
| 3 | 0.0592 | 0.0881 | 0.1648 | 0.0746 |
| 4 | 0.1648 | 0.1592 | 0.3777 | 0.3125 |
| 5 | 0.0933 | 0.1040 | 0.2326 | 0.2688 |
| **Mean** | **0.1190** | **0.1412** | **0.2590** | **0.2335** |

## Per-Station Out-of-Fold Performance

| Station | n | Positives | Base PR-AUC | Full PR-AUC | ΔPR-AUC | Base F2 | Full F2 |
|---------|---|-----------|-------------|-------------|---------|---------|---------|
| hafar | 911 | 50 | 0.1415 | 0.1457 | +0.0042 | 0.3303 | 0.3134 |
| riyadh | 912 | 42 | 0.1188 | 0.1226 | +0.0038 | 0.2742 | 0.2623 |
| sharurah | 912 | 34 | 0.0924 | 0.1057 | +0.0133 | 0.1923 | 0.1337 |

## Driver Ablation — What Actually Matters

Incremental PR-AUC of each physical driver group (full model − model without that group), with paired bootstrap 95% CIs. A CI entirely above zero means the driver contributes skill no other group supplies.

| Driver group | # feats | Incremental PR-AUC | 95% CI | Significant |
|--------------|---------|--------------------|--------|-------------|
| vegetation | 1 | +0.0183 | [+0.0065, +0.0371] | **yes** |
| antecedent_moisture | 8 | +0.0079 | [-0.0037, +0.0211] | no |
| seasonality | 4 | +0.0078 | [-0.0038, +0.0209] | no |
| pressure | 1 | +0.0069 | [-0.0021, +0.0180] | no |
| wind_direction | 3 | +0.0051 | [-0.0184, +0.0213] | no |
| wind_speed | 13 | +0.0046 | [-0.0229, +0.0268] | no |
| albedo | 4 | +0.0040 | [-0.0145, +0.0224] | no |
| thermal_blh | 8 | +0.0011 | [-0.0118, +0.0147] | no |
| humidity_dryness | 10 | -0.0018 | [-0.0179, +0.0121] | no |
| soil_texture | 4 | -0.0026 | [-0.0130, +0.0069] | no |

**Significant drivers of next-day dust:** vegetation.

## Statistical Tests

- **Paired bootstrap (5 000 resamples)** on out-of-fold predictions gives the 95% CIs in the headline table — the primary inference.
- **Wilcoxon signed-rank on per-fold PR-AUC** (n=5): W=4.00, p=0.4375 (not significant at α=0.05).
- **Wilcoxon signed-rank on per-fold F2** (n=5): W=3.00, p=0.3125 (not significant at α=0.05).

## Figures

- `pr_curves.png` — precision–recall curves (baseline vs full)
- `f2_comparison_by_fold.png` — per-fold F2 bar chart
- `bootstrap_delta_f2.png` — bootstrap ΔF2 distribution
- `shap_importance.png` — SHAP feature importance (full model)

## Conclusion

Adding the MODIS albedo anomaly yields a positive but not statistically conclusive PR-AUC gain (+0.0040, 95% CI [-0.0147, +0.0230] includes zero): suggestive evidence that surface reflectivity adds incremental dust-forecast skill, warranting a larger sample / wider MODIS footprint.
