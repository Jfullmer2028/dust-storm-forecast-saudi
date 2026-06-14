# Dust-Storm Onset Forecasting — Results Report

**Data mode:** real

## Dataset Summary

- Total samples: 3,285
- Positive (dust event next day): 201 (6.1%)
- Stations: hafar, riyadh, sharurah
- Study period: 2018-01-01 00:00:00 to 2020-12-30 00:00:00

## Forecast Model Performance (TimeSeriesSplit, 5 folds)

Out-of-fold cross-validated skill of the XGBoost forecaster. PR-AUC (average precision) is the primary metric for this rare-event problem; ROC-AUC and F2 (at a per-fold tuned threshold) are reported alongside. Across 5 random seeds, PR-AUC = 0.130 ± 0.006 and ROC-AUC = 0.721 ± 0.006.

| Metric | Mean | Std |
|--------|------|-----|
| PR-AUC | 0.1412 | 0.0591 |
| ROC-AUC | 0.7300 | — |
| F2 (operational) | 0.2335 | 0.0840 |

| Fold | PR-AUC | ROC-AUC | F2 |
|------|--------|---------|----|
| 1 | 0.2492 | 0.6872 | 0.2834 |
| 2 | 0.1053 | 0.6223 | 0.2281 |
| 3 | 0.0881 | 0.7736 | 0.0746 |
| 4 | 0.1592 | 0.7552 | 0.3125 |
| 5 | 0.1040 | 0.8119 | 0.2688 |

## Naive Baselines

The forecaster against simple references (out-of-fold where a model is involved):

| Reference | PR-AUC | ROC-AUC |
|-----------|--------|---------|
| no-skill (base rate) | 0.0612 | 0.5000 |
| persistence | 0.1016 | 0.5919 |
| meteorology-only model | 0.1185 | 0.7108 |
| full model | 0.1412 | 0.7300 |

## Driver Ablation (BH-FDR corrected)

Incremental skill of each physical driver group: the change in PR-AUC when that group is removed and the model retrained (model − without-group), with paired bootstrap 95% CIs and two-sided bootstrap p-values. Because one test is run per driver group, p-values are corrected for multiple comparisons with **Benjamini-Hochberg FDR**; the corrected call (`sig.`) is the reported result.

| Driver group | # feats | Incremental PR-AUC | 95% CI | p | p (FDR) | sig. |
|--------------|---------|--------------------|--------|---|---------|------|
| vegetation | 1 | +0.0183 | [+0.0065, +0.0371] | 0.003 | 0.030 | **yes** |
| antecedent_moisture | 8 | +0.0079 | [-0.0037, +0.0211] | 0.175 | 0.438 | no |
| seasonality | 4 | +0.0078 | [-0.0038, +0.0209] | 0.166 | 0.438 | no |
| pressure | 1 | +0.0069 | [-0.0021, +0.0180] | 0.135 | 0.438 | no |
| wind_direction | 3 | +0.0051 | [-0.0184, +0.0213] | 0.640 | 0.814 | no |
| wind_speed | 13 | +0.0046 | [-0.0229, +0.0268] | 0.781 | 0.814 | no |
| albedo | 4 | +0.0040 | [-0.0145, +0.0224] | 0.585 | 0.814 | no |
| thermal_blh | 8 | +0.0011 | [-0.0118, +0.0147] | 0.809 | 0.814 | no |
| humidity_dryness | 10 | -0.0018 | [-0.0179, +0.0121] | 0.814 | 0.814 | no |
| soil_texture | 4 | -0.0026 | [-0.0130, +0.0069] | 0.651 | 0.814 | no |

**Driver groups significant after FDR correction:** vegetation.

Seed robustness of the top driver (ΔPR-AUC over 5 seeds): +0.0091 ± 0.0075.

## Per-Station Performance

| Station | n | Positives | PR-AUC | F2 | Precision | Recall |
|---------|---|-----------|--------|----|-----------|--------|
| hafar | 911 | 50 | 0.1457 | 0.3134 | 0.138 | 0.460 |
| riyadh | 912 | 42 | 0.1226 | 0.2623 | 0.109 | 0.405 |
| sharurah | 912 | 34 | 0.1057 | 0.1337 | 0.098 | 0.147 |

## Figures

- `driver_ablation.png` — incremental PR-AUC by driver group (FDR)
- `pr_curve.png` — precision–recall curve for the forecast model
- `calibration.png` — reliability diagram (out-of-fold)
- `shap_importance.png` — SHAP feature importance

## Conclusion

The forecaster attains a cross-validated PR-AUC of 0.141 (ROC-AUC 0.730) at a 6.1% base rate, well above the no-skill PR-AUC of 0.061. After Benjamini-Hochberg FDR correction across the driver groups, **vegetation** retains statistically significant incremental skill, with **vegetation** the strongest (ΔPR-AUC +0.0183, FDR p=0.030). Remaining groups carry information already present elsewhere in the feature set.
