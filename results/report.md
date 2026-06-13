# Dust-Storm Onset Prediction — Results Report

**Data mode:** synthetic
**Generated:** pipeline run

## Dataset Summary

- Total samples: 14,576
- Positive (dust event next day): 1,250 (8.6%)
- Stations: arar, dammam, hafar, najran, qassim, riyadh, sharurah, tabuk
- Study period: 2018-01-01 00:00:00 to 2022-12-27 00:00:00

## Headline Comparison

Primary metric is **PR-AUC (average precision)** — threshold-independent and appropriate for this rare-event problem. ROC-AUC is reported alongside it; F2 at the per-fold tuned threshold is the **operational** metric. All deltas are full − baseline with paired bootstrap 95% CIs (out-of-fold predictions, 8-fold TimeSeriesSplit).

| Metric | Baseline | Full | Δ | 95% CI | Verdict |
|--------|----------|------|---|--------|---------|
| **PR-AUC** (primary) | 0.3633 | 0.4244 | +0.0611 | [+0.0446, +0.0779] | CI entirely above 0 — full model significantly better |
| ROC-AUC | 0.8172 | 0.8658 | +0.0487 | [+0.0406, +0.0567] | CI entirely above 0 — full model significantly better |
| F2 @ tuned thr (operational) | 0.4806 | 0.5571 | +0.0765 | [+0.0551, +0.0858] | CI entirely above 0 — full model significantly better |

## Per-Fold Cross-Validation (TimeSeriesSplit, 8 folds)

| Fold | Baseline PR-AUC | Full PR-AUC | Baseline F2 | Full F2 |
|------|-----------------|-------------|-------------|---------|
| 1 | 0.3399 | 0.3894 | 0.4560 | 0.5112 |
| 2 | 0.4178 | 0.4159 | 0.4703 | 0.5268 |
| 3 | 0.4175 | 0.5247 | 0.4776 | 0.5569 |
| 4 | 0.3762 | 0.4271 | 0.5402 | 0.5763 |
| 5 | 0.3154 | 0.4347 | 0.3874 | 0.5540 |
| 6 | 0.4070 | 0.4861 | 0.5056 | 0.5929 |
| 7 | 0.2823 | 0.3449 | 0.4613 | 0.5168 |
| 8 | 0.3966 | 0.4524 | 0.5462 | 0.6216 |
| **Mean** | **0.3691** | **0.4344** | **0.4806** | **0.5571** |

## Per-Station Out-of-Fold Performance

| Station | n | Positives | Base PR-AUC | Full PR-AUC | ΔPR-AUC | Base F2 | Full F2 |
|---------|---|-----------|-------------|-------------|---------|---------|---------|
| arar | 1619 | 110 | 0.2495 | 0.3514 | +0.1019 | 0.4281 | 0.5144 |
| dammam | 1619 | 123 | 0.3086 | 0.4124 | +0.1038 | 0.4292 | 0.5621 |
| hafar | 1619 | 175 | 0.4797 | 0.5153 | +0.0355 | 0.5497 | 0.5946 |
| najran | 1619 | 192 | 0.3980 | 0.4354 | +0.0374 | 0.5356 | 0.5982 |
| qassim | 1619 | 119 | 0.4025 | 0.4362 | +0.0337 | 0.4694 | 0.5193 |
| riyadh | 1619 | 143 | 0.3653 | 0.4508 | +0.0856 | 0.5113 | 0.5757 |
| sharurah | 1619 | 103 | 0.2641 | 0.2919 | +0.0277 | 0.4743 | 0.5193 |
| tabuk | 1619 | 134 | 0.3965 | 0.4757 | +0.0792 | 0.4704 | 0.5538 |

## Driver Ablation — What Actually Matters

Incremental PR-AUC of each physical driver group (full model − model without that group), with paired bootstrap 95% CIs. A CI entirely above zero means the driver contributes skill no other group supplies.

| Driver group | # feats | Incremental PR-AUC | 95% CI | Significant |
|--------------|---------|--------------------|--------|-------------|
| wind_direction | 3 | +0.0699 | [+0.0508, +0.0887] | **yes** |
| albedo | 4 | +0.0611 | [+0.0441, +0.0777] | **yes** |
| wind_speed | 18 | +0.0112 | [-0.0001, +0.0224] | no |
| soil_texture | 5 | +0.0094 | [-0.0008, +0.0204] | no |
| humidity_dryness | 7 | +0.0058 | [-0.0057, +0.0173] | no |
| thermal_blh | 14 | +0.0055 | [-0.0056, +0.0172] | no |
| pressure | 1 | +0.0005 | [-0.0082, +0.0097] | no |
| antecedent_moisture | 8 | -0.0007 | [-0.0127, +0.0117] | no |
| vegetation | 1 | -0.0065 | [-0.0153, +0.0024] | no |
| seasonality | 4 | -0.0075 | [-0.0169, +0.0018] | no |

**Significant drivers of next-day dust:** wind_direction, albedo.

## Statistical Tests

- **Paired bootstrap (5 000 resamples)** on out-of-fold predictions gives the 95% CIs in the headline table — the primary inference.
- **Wilcoxon signed-rank on per-fold PR-AUC** (n=8): W=1.00, p=0.0156 (significant at α=0.05).
- **Wilcoxon signed-rank on per-fold F2** (n=8): W=0.00, p=0.0078 (significant at α=0.05).

## Figures

- `pr_curves.png` — precision–recall curves (baseline vs full)
- `f2_comparison_by_fold.png` — per-fold F2 bar chart
- `bootstrap_delta_f2.png` — bootstrap ΔF2 distribution
- `shap_importance.png` — SHAP feature importance (full model)

## Conclusion

On the primary threshold-independent metric (PR-AUC), adding the MODIS shortwave broadband albedo anomaly **significantly improves** 24-hour dust-storm risk ranking: the paired bootstrap 95% CI for ΔPR-AUC is entirely above zero (+0.0611 [+0.0446, +0.0779]).
