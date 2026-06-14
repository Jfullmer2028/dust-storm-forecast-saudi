# Dust-Storm Onset Forecasting — Results Report

**Data mode:** synthetic

## Dataset Summary

- Total samples: 14,576
- Positive (dust event next day): 1,260 (8.6%)
- Stations: arar, dammam, hafar, najran, qassim, riyadh, sharurah, tabuk
- Study period: 2018-01-01 00:00:00 to 2022-12-27 00:00:00

## Forecast Model Performance (TimeSeriesSplit, 8 folds)

Out-of-fold cross-validated skill of the XGBoost forecaster. PR-AUC (average precision) is the primary metric for this rare-event problem; ROC-AUC and F2 (at a per-fold tuned threshold) are reported alongside.

| Metric | Mean | Std |
|--------|------|-----|
| PR-AUC | 0.3794 | 0.0563 |
| ROC-AUC | 0.8455 | — |
| F2 (operational) | 0.5237 | 0.0345 |

| Fold | PR-AUC | ROC-AUC | F2 |
|------|--------|---------|----|
| 1 | 0.3247 | 0.7915 | 0.4614 |
| 2 | 0.4248 | 0.8306 | 0.5026 |
| 3 | 0.4185 | 0.8874 | 0.5319 |
| 4 | 0.3888 | 0.8256 | 0.5665 |
| 5 | 0.3853 | 0.8728 | 0.5172 |
| 6 | 0.4158 | 0.8463 | 0.5591 |
| 7 | 0.2549 | 0.8362 | 0.4939 |
| 8 | 0.4222 | 0.8739 | 0.5571 |

## Driver Ablation

Incremental skill of each physical driver group: the change in PR-AUC when that group is removed and the model retrained (model − without-group), with paired bootstrap 95% CIs on the out-of-fold predictions. A CI entirely above zero marks a driver that carries information not already present in the other features.

| Driver group | # feats | Incremental PR-AUC | 95% CI | Significant |
|--------------|---------|--------------------|--------|-------------|
| wind_direction | 3 | +0.0505 | [+0.0326, +0.0681] | **yes** |
| vegetation | 1 | +0.0159 | [+0.0034, +0.0278] | **yes** |
| antecedent_moisture | 8 | +0.0059 | [-0.0047, +0.0161] | no |
| thermal_blh | 14 | +0.0055 | [-0.0049, +0.0167] | no |
| humidity_dryness | 7 | +0.0045 | [-0.0062, +0.0149] | no |
| seasonality | 4 | +0.0042 | [-0.0042, +0.0124] | no |
| soil_texture | 5 | +0.0036 | [-0.0054, +0.0130] | no |
| wind_speed | 18 | +0.0024 | [-0.0089, +0.0142] | no |
| pressure | 1 | -0.0015 | [-0.0097, +0.0073] | no |
| albedo | 4 | -0.0049 | [-0.0146, +0.0043] | no |

**Driver groups with significant incremental skill:** wind_direction, vegetation.

## Per-Station Performance

| Station | n | Positives | PR-AUC | F2 | Precision | Recall |
|---------|---|-----------|--------|----|-----------|--------|
| arar | 1619 | 114 | 0.3333 | 0.5155 | 0.250 | 0.702 |
| dammam | 1619 | 125 | 0.3088 | 0.5075 | 0.240 | 0.704 |
| hafar | 1619 | 177 | 0.4701 | 0.5522 | 0.280 | 0.729 |
| najran | 1619 | 192 | 0.3757 | 0.5677 | 0.285 | 0.755 |
| qassim | 1619 | 122 | 0.4047 | 0.5078 | 0.223 | 0.746 |
| riyadh | 1619 | 141 | 0.3750 | 0.5460 | 0.254 | 0.766 |
| sharurah | 1619 | 105 | 0.2825 | 0.4605 | 0.206 | 0.667 |
| tabuk | 1619 | 133 | 0.4112 | 0.5351 | 0.252 | 0.744 |

## Figures

- `driver_ablation.png` — incremental PR-AUC by driver group
- `pr_curve.png` — precision–recall curve for the forecast model
- `shap_importance.png` — SHAP feature importance

## Conclusion

The forecaster attains a cross-validated PR-AUC of 0.379 (ROC-AUC 0.846) at a 8.6% base rate. The driver ablation identifies **wind_direction, vegetation** as the feature groups contributing statistically significant incremental skill, with **wind_direction** the strongest (ΔPR-AUC +0.0505, 95% CI [+0.0326, +0.0681]). Remaining groups carry information already present elsewhere in the feature set.
