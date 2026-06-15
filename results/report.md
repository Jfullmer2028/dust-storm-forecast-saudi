# Dust-Storm Onset Forecasting — Results Report

**Data mode:** synthetic

## Dataset Summary

- Total samples: 14,576
- Positive (dust event next day): 1,260 (8.6%)
- Stations: arar, dammam, hafar, najran, qassim, riyadh, sharurah, tabuk
- Study period: 2018-01-01 00:00:00 to 2022-12-27 00:00:00

## Forecast Model Performance (TimeSeriesSplit, 8 folds)

Out-of-fold cross-validated skill of the XGBoost forecaster. PR-AUC (average precision) is the primary metric for this rare-event problem; ROC-AUC and F2 (at a per-fold tuned threshold) are reported alongside. Across 5 random seeds, PR-AUC = 0.378 ± 0.004 and ROC-AUC = 0.843 ± 0.001.

| Metric | Mean | Std |
|--------|------|-----|
| PR-AUC | 0.3706 | 0.0580 |
| ROC-AUC | 0.8413 | — |
| F2 (operational) | 0.5013 | 0.0596 |

| Fold | PR-AUC | ROC-AUC | F2 |
|------|--------|---------|----|
| 1 | 0.3094 | 0.7836 | 0.4132 |
| 2 | 0.3920 | 0.8236 | 0.4905 |
| 3 | 0.4268 | 0.8864 | 0.5120 |
| 4 | 0.3900 | 0.8240 | 0.5644 |
| 5 | 0.3865 | 0.8676 | 0.5020 |
| 6 | 0.4103 | 0.8477 | 0.5501 |
| 7 | 0.2438 | 0.8251 | 0.4062 |
| 8 | 0.4063 | 0.8724 | 0.5720 |

## Naive Baselines

The forecaster against simple references (out-of-fold where a model is involved):

| Reference | PR-AUC | ROC-AUC |
|-----------|--------|---------|
| no-skill (base rate) | 0.0864 | 0.5000 |
| persistence | 0.2554 | 0.7025 |
| meteorology-only model | 0.3570 | 0.8153 |
| full model | 0.3706 | 0.8413 |

## Operational Evaluation

Judging the probabilities as a warning system (out-of-fold):

| Quantity | Value |
|----------|-------|
| Brier score | 0.0776 |
| Brier Skill Score (vs climatology) | +0.009 |
| Recall at precision ≥ 0.30 | 0.58 |
| Precision at recall = 0.50 | 0.34 |
| False-alarm rate at recall = 0.50 | 0.09 |

To catch **half of all dust days**, the model issues warnings on 9% of calm days (precision 0.34 at a 8.6% base rate — a 4.0× lift over random). The positive Brier Skill Score means the calibrated probabilities are sharper than a climatology forecast.

## Driver Ablation (BH-FDR corrected)

Incremental skill of each physical driver group: the change in PR-AUC when that group is removed and the model retrained (model − without-group), with paired bootstrap 95% CIs and two-sided bootstrap p-values. Because one test is run per driver group, p-values are corrected for multiple comparisons with **Benjamini-Hochberg FDR**; the corrected call (`sig.`) is the reported result.

| Driver group | # feats | Incremental PR-AUC | 95% CI | p | p (FDR) | sig. |
|--------------|---------|--------------------|--------|---|---------|------|
| wind_direction | 3 | +0.0506 | [+0.0337, +0.0678] | 0.001 | 0.005 | **yes** |
| vegetation | 1 | +0.0118 | [-0.0007, +0.0252] | 0.067 | 0.198 | no |
| thermal_blh | 14 | +0.0013 | [-0.0098, +0.0124] | 0.835 | 0.835 | no |
| soil_texture | 5 | -0.0013 | [-0.0112, +0.0080] | 0.757 | 0.835 | no |
| antecedent_moisture | 10 | -0.0017 | [-0.0135, +0.0093] | 0.727 | 0.835 | no |
| wind_speed | 18 | -0.0027 | [-0.0146, +0.0089] | 0.634 | 0.835 | no |
| pressure | 1 | -0.0039 | [-0.0129, +0.0045] | 0.331 | 0.643 | no |
| humidity_dryness | 7 | -0.0049 | [-0.0154, +0.0061] | 0.386 | 0.643 | no |
| seasonality | 4 | -0.0079 | [-0.0165, +0.0002] | 0.055 | 0.198 | no |
| albedo | 4 | -0.0084 | [-0.0186, +0.0012] | 0.079 | 0.198 | no |

**Driver groups significant after FDR correction:** wind_direction.

Seed robustness of the top driver (ΔPR-AUC over 5 seeds): +0.0523 ± 0.0022.

## Per-Station Performance

| Station | n | Positives | PR-AUC | F2 | Precision | Recall |
|---------|---|-----------|--------|----|-----------|--------|
| arar | 1619 | 114 | 0.3205 | 0.4738 | 0.220 | 0.667 |
| dammam | 1619 | 125 | 0.2980 | 0.4571 | 0.213 | 0.640 |
| hafar | 1619 | 177 | 0.4611 | 0.5469 | 0.280 | 0.718 |
| najran | 1619 | 192 | 0.3588 | 0.5460 | 0.272 | 0.729 |
| qassim | 1619 | 122 | 0.3986 | 0.4911 | 0.216 | 0.721 |
| riyadh | 1619 | 141 | 0.3782 | 0.5213 | 0.243 | 0.730 |
| sharurah | 1619 | 105 | 0.2831 | 0.4696 | 0.211 | 0.676 |
| tabuk | 1619 | 133 | 0.4023 | 0.5430 | 0.254 | 0.759 |

## Figures

- `driver_ablation.png` — incremental PR-AUC by driver group (FDR)
- `pr_curve.png` — precision–recall curve for the forecast model
- `calibration.png` — reliability diagram (out-of-fold)
- `shap_importance.png` — SHAP feature importance

## Conclusion

The forecaster attains a cross-validated PR-AUC of 0.371 (ROC-AUC 0.841) at a 8.6% base rate, well above the no-skill PR-AUC of 0.086. After Benjamini-Hochberg FDR correction across the driver groups, **wind_direction** retains statistically significant incremental skill, with **wind_direction** the strongest (ΔPR-AUC +0.0506, FDR p=0.005). Remaining groups carry information already present elsewhere in the feature set.
