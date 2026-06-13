# Dust-Storm Onset Prediction — Results Report

**Data mode:** synthetic
**Generated:** pipeline run

## Dataset Summary

- Total samples: 14,576
- Positive (dust event next day): 1,239 (8.5%)
- Stations: arar, dammam, hafar, najran, qassim, riyadh, sharurah, tabuk
- Study period: 2018-01-01 00:00:00 to 2022-12-27 00:00:00

## Headline Comparison

Primary metric is **PR-AUC (average precision)** — threshold-independent and appropriate for this rare-event problem. ROC-AUC is reported alongside it; F2 at the per-fold tuned threshold is the **operational** metric. All deltas are full − baseline with paired bootstrap 95% CIs (out-of-fold predictions, 8-fold TimeSeriesSplit).

| Metric | Baseline | Full | Δ | 95% CI | Verdict |
|--------|----------|------|---|--------|---------|
| **PR-AUC** (primary) | 0.3238 | 0.3973 | +0.0735 | [+0.0539, +0.0931] | CI entirely above 0 — full model significantly better |
| ROC-AUC | 0.8097 | 0.8633 | +0.0536 | [+0.0452, +0.0619] | CI entirely above 0 — full model significantly better |
| F2 @ tuned thr (operational) | 0.4764 | 0.5589 | +0.0825 | [+0.0623, +0.0965] | CI entirely above 0 — full model significantly better |

## Per-Fold Cross-Validation (TimeSeriesSplit, 8 folds)

| Fold | Baseline PR-AUC | Full PR-AUC | Baseline F2 | Full F2 |
|------|-----------------|-------------|-------------|---------|
| 1 | 0.2559 | 0.3296 | 0.4627 | 0.4898 |
| 2 | 0.3248 | 0.3630 | 0.4815 | 0.5269 |
| 3 | 0.3417 | 0.4591 | 0.4599 | 0.5919 |
| 4 | 0.3342 | 0.4155 | 0.4662 | 0.5776 |
| 5 | 0.2553 | 0.3922 | 0.3795 | 0.5077 |
| 6 | 0.4215 | 0.4987 | 0.5585 | 0.6276 |
| 7 | 0.2863 | 0.3290 | 0.4510 | 0.5635 |
| 8 | 0.4023 | 0.4451 | 0.5517 | 0.5861 |
| **Mean** | **0.3278** | **0.4040** | **0.4764** | **0.5589** |

## Per-Station Out-of-Fold Performance

| Station | n | Positives | Base PR-AUC | Full PR-AUC | ΔPR-AUC | Base F2 | Full F2 |
|---------|---|-----------|-------------|-------------|---------|---------|---------|
| arar | 1619 | 115 | 0.2439 | 0.3806 | +0.1367 | 0.4089 | 0.5153 |
| dammam | 1619 | 127 | 0.2967 | 0.4106 | +0.1139 | 0.4691 | 0.5361 |
| hafar | 1619 | 173 | 0.4148 | 0.4446 | +0.0298 | 0.5230 | 0.6018 |
| najran | 1619 | 192 | 0.3507 | 0.3945 | +0.0439 | 0.5472 | 0.6270 |
| qassim | 1619 | 124 | 0.3271 | 0.4023 | +0.0752 | 0.4769 | 0.5347 |
| riyadh | 1619 | 132 | 0.3882 | 0.5011 | +0.1129 | 0.4611 | 0.5805 |
| sharurah | 1619 | 107 | 0.2627 | 0.3015 | +0.0389 | 0.4771 | 0.5072 |
| tabuk | 1619 | 132 | 0.3069 | 0.3634 | +0.0564 | 0.4431 | 0.5319 |

## Statistical Tests

- **Paired bootstrap (5 000 resamples)** on out-of-fold predictions gives the 95% CIs in the headline table — the primary inference.
- **Wilcoxon signed-rank on per-fold PR-AUC** (n=8): W=0.00, p=0.0078 (significant at α=0.05).
- **Wilcoxon signed-rank on per-fold F2** (n=8): W=0.00, p=0.0078 (significant at α=0.05).

## Figures

- `pr_curves.png` — precision–recall curves (baseline vs full)
- `f2_comparison_by_fold.png` — per-fold F2 bar chart
- `bootstrap_delta_f2.png` — bootstrap ΔF2 distribution
- `shap_importance.png` — SHAP feature importance (full model)

## Conclusion

On the primary threshold-independent metric (PR-AUC), adding the MODIS shortwave broadband albedo anomaly **significantly improves** 24-hour dust-storm risk ranking: the paired bootstrap 95% CI for ΔPR-AUC is entirely above zero (+0.0735 [+0.0539, +0.0931]).
