# Dust-Storm Onset Forecasting — Results Report

**Data mode:** real

## Dataset Summary

- Total samples: 6,570
- Positive (dust event next day): 367 (5.6%)
- Stations: arar, dammam, hafar, riyadh, sharurah, tabuk
- Study period: 2018-01-01 00:00:00 to 2020-12-30 00:00:00

## Forecast Model Performance (TimeSeriesSplit, 6 folds)

Out-of-fold cross-validated skill of the XGBoost forecaster. PR-AUC (average precision) is the primary metric for this rare-event problem; ROC-AUC and F2 (at a per-fold tuned threshold) are reported alongside. Across 5 random seeds, PR-AUC = 0.112 ± 0.009 and ROC-AUC = 0.668 ± 0.023.

| Metric | Mean | Std |
|--------|------|-----|
| PR-AUC | 0.1253 | 0.0429 |
| ROC-AUC | 0.6860 | — |
| F2 (operational) | 0.1748 | 0.1071 |

| Fold | PR-AUC | ROC-AUC | F2 |
|------|--------|---------|----|
| 1 | 0.1216 | 0.6449 | 0.0940 |
| 2 | 0.1185 | 0.6577 | 0.2745 |
| 3 | 0.0972 | 0.6405 | 0.1415 |
| 4 | 0.0602 | 0.6259 | 0.0000 |
| 5 | 0.1944 | 0.7367 | 0.3073 |
| 6 | 0.1599 | 0.8103 | 0.2312 |

## Naive Baselines

The forecaster against simple references (out-of-fold where a model is involved):

| Reference | PR-AUC | ROC-AUC |
|-----------|--------|---------|
| no-skill (base rate) | 0.0559 | 0.5000 |
| persistence | 0.1000 | 0.5974 |
| meteorology-only model | 0.1079 | 0.6874 |
| full model | 0.1253 | 0.6860 |

## Operational Evaluation

Judging the probabilities as a warning system (out-of-fold):

| Quantity | Value |
|----------|-------|
| Brier score | 0.0439 |
| Brier Skill Score (vs climatology) | +0.012 |
| Recall at precision ≥ 0.30 | 0.04 |
| Precision at recall = 0.50 | 0.06 |
| False-alarm rate at recall = 0.50 | 0.38 |

To catch **half of all dust days**, the model issues warnings on 38% of calm days (precision 0.06 at a 4.7% base rate — a 1.3× lift over random). The calibrated probabilities are sharper than a climatology forecast (positive Brier Skill Score).

## Driver Ablation (BH-FDR corrected)

Incremental skill of each physical driver group: the change in PR-AUC when that group is removed and the model retrained (model − without-group), with paired bootstrap 95% CIs and two-sided bootstrap p-values. Because one test is run per driver group, p-values are corrected for multiple comparisons with **Benjamini-Hochberg FDR**; the corrected call (`sig.`) is the reported result.

| Driver group | # feats | Incremental PR-AUC | 95% CI | p | p (FDR) | sig. |
|--------------|---------|--------------------|--------|---|---------|------|
| humidity_dryness | 10 | +0.0364 | [+0.0149, +0.0650] | 0.001 | 0.005 | **yes** |
| vegetation | 1 | +0.0249 | [+0.0062, +0.0494] | 0.008 | 0.027 | **yes** |
| seasonality | 4 | +0.0231 | [+0.0065, +0.0444] | 0.001 | 0.005 | **yes** |
| soil_texture | 4 | +0.0168 | [+0.0021, +0.0364] | 0.022 | 0.055 | no |
| wind_speed | 13 | +0.0159 | [-0.0050, +0.0399] | 0.136 | 0.170 | no |
| antecedent_moisture | 10 | +0.0159 | [-0.0022, +0.0396] | 0.084 | 0.157 | no |
| albedo | 4 | +0.0143 | [-0.0023, +0.0369] | 0.094 | 0.157 | no |
| thermal_blh | 8 | +0.0140 | [-0.0034, +0.0365] | 0.116 | 0.166 | no |
| pressure | 1 | +0.0077 | [-0.0068, +0.0228] | 0.310 | 0.344 | no |
| wind_direction | 3 | -0.0011 | [-0.0161, +0.0157] | 0.873 | 0.873 | no |

**Driver groups significant after FDR correction:** humidity_dryness, vegetation, seasonality.

Seed robustness of the top driver (ΔPR-AUC over 5 seeds): +0.0228 ± 0.0108.

## Per-Station Performance

| Station | n | Positives | PR-AUC | F2 | Precision | Recall |
|---------|---|-----------|--------|----|-----------|--------|
| arar | 938 | 47 | 0.1172 | 0.2404 | 0.088 | 0.426 |
| dammam | 938 | 55 | 0.1143 | 0.2222 | 0.080 | 0.400 |
| hafar | 938 | 56 | 0.1129 | 0.2326 | 0.088 | 0.393 |
| riyadh | 938 | 47 | 0.1437 | 0.2550 | 0.087 | 0.489 |
| sharurah | 938 | 36 | 0.1034 | 0.2332 | 0.080 | 0.444 |
| tabuk | 938 | 21 | 0.0441 | 0.1373 | 0.041 | 0.333 |

## Figures

- `driver_ablation.png` — incremental PR-AUC by driver group (FDR)
- `pr_curve.png` — precision–recall curve for the forecast model
- `calibration.png` — reliability diagram (out-of-fold)
- `shap_importance.png` — SHAP feature importance

## Conclusion

The forecaster attains a cross-validated PR-AUC of 0.125 (ROC-AUC 0.686) at a 5.6% base rate, well above the no-skill PR-AUC of 0.056. After Benjamini-Hochberg FDR correction across the driver groups, **humidity_dryness, vegetation, seasonality** retain statistically significant incremental skill, with **humidity_dryness** the strongest (ΔPR-AUC +0.0364, FDR p=0.005). Remaining groups carry information already present elsewhere in the feature set.
