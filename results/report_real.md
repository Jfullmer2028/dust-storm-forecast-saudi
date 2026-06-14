# Dust-Storm Onset Forecasting — Results Report

**Data mode:** real

## Dataset Summary

- Total samples: 3,285
- Positive (dust event next day): 201 (6.1%)
- Stations: hafar, riyadh, sharurah
- Study period: 2018-01-01 00:00:00 to 2020-12-30 00:00:00

## Forecast Model Performance (TimeSeriesSplit, 5 folds)

Out-of-fold cross-validated skill of the XGBoost forecaster. PR-AUC (average precision) is the primary metric for this rare-event problem; ROC-AUC and F2 (at a per-fold tuned threshold) are reported alongside.

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

## Driver Ablation

Incremental skill of each physical driver group: the change in PR-AUC when that group is removed and the model retrained (model − without-group), with paired bootstrap 95% CIs on the out-of-fold predictions. A CI entirely above zero marks a driver that carries information not already present in the other features.

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

**Driver groups with significant incremental skill:** vegetation.

## Per-Station Performance

| Station | n | Positives | PR-AUC | F2 | Precision | Recall |
|---------|---|-----------|--------|----|-----------|--------|
| hafar | 911 | 50 | 0.1457 | 0.3134 | 0.138 | 0.460 |
| riyadh | 912 | 42 | 0.1226 | 0.2623 | 0.109 | 0.405 |
| sharurah | 912 | 34 | 0.1057 | 0.1337 | 0.098 | 0.147 |

## Figures

- `driver_ablation.png` — incremental PR-AUC by driver group
- `pr_curve.png` — precision–recall curve for the forecast model
- `shap_importance.png` — SHAP feature importance

## Conclusion

The forecaster attains a cross-validated PR-AUC of 0.141 (ROC-AUC 0.730) at a 6.1% base rate. The driver ablation identifies **vegetation** as the only feature group contributing statistically significant incremental skill (ΔPR-AUC +0.0183, 95% CI [+0.0065, +0.0371]). Remaining groups carry information already present elsewhere in the feature set.
