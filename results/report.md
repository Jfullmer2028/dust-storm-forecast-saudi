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
| F2 (operational) | 0.5061 | 0.0577 |

| Fold | PR-AUC | ROC-AUC | F2 |
|------|--------|---------|----|
| 1 | 0.3094 | 0.7836 | 0.4064 |
| 2 | 0.3920 | 0.8236 | 0.4954 |
| 3 | 0.4268 | 0.8864 | 0.5387 |
| 4 | 0.3900 | 0.8240 | 0.5475 |
| 5 | 0.3865 | 0.8676 | 0.5034 |
| 6 | 0.4103 | 0.8477 | 0.5582 |
| 7 | 0.2438 | 0.8251 | 0.4254 |
| 8 | 0.4063 | 0.8724 | 0.5739 |

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
| Brier score | 0.0673 |
| Brier Skill Score (vs climatology) | +0.141 |
| Recall at precision ≥ 0.30 | 0.56 |
| Precision at recall = 0.50 | 0.34 |
| False-alarm rate at recall = 0.50 | 0.09 |

To catch **half of all dust days**, the model issues warnings on 9% of calm days (precision 0.34 at a 8.6% base rate — a 4.0× lift over random). The calibrated probabilities are sharper than a climatology forecast (positive Brier Skill Score).

## Driver Ablation (BH-FDR corrected)

Incremental skill of each physical driver group: the change in PR-AUC when that group is removed and the model retrained (model − without-group), with paired bootstrap 95% CIs and two-sided bootstrap p-values. Because one test is run per driver group, p-values are corrected for multiple comparisons with **Benjamini-Hochberg FDR**; the corrected call (`sig.`) is the reported result.

| Driver group | # feats | Incremental PR-AUC | 95% CI | p | p (FDR) | sig. |
|--------------|---------|--------------------|--------|---|---------|------|
| wind_direction | 3 | +0.0523 | [+0.0364, +0.0679] | 0.001 | 0.005 | **yes** |
| vegetation | 1 | +0.0168 | [+0.0028, +0.0313] | 0.017 | 0.085 | no |
| wind_speed | 18 | +0.0050 | [-0.0074, +0.0164] | 0.485 | 0.902 | no |
| pressure | 1 | +0.0028 | [-0.0057, +0.0114] | 0.575 | 0.902 | no |
| antecedent_moisture | 10 | +0.0023 | [-0.0092, +0.0132] | 0.722 | 0.902 | no |
| thermal_blh | 14 | +0.0021 | [-0.0090, +0.0143] | 0.703 | 0.902 | no |
| soil_texture | 5 | -0.0006 | [-0.0113, +0.0097] | 0.902 | 0.902 | no |
| humidity_dryness | 7 | -0.0010 | [-0.0124, +0.0102] | 0.816 | 0.902 | no |
| seasonality | 4 | -0.0054 | [-0.0144, +0.0036] | 0.260 | 0.650 | no |
| albedo | 4 | -0.0080 | [-0.0184, +0.0022] | 0.122 | 0.407 | no |

**Driver groups significant after FDR correction:** wind_direction.

Seed robustness of the top driver (ΔPR-AUC over 5 seeds): +0.0523 ± 0.0022.

## Per-Station Performance

| Station | n | Positives | PR-AUC | F2 | Precision | Recall |
|---------|---|-----------|--------|----|-----------|--------|
| arar | 1619 | 114 | 0.3031 | 0.4914 | 0.223 | 0.702 |
| dammam | 1619 | 125 | 0.2584 | 0.4824 | 0.223 | 0.680 |
| hafar | 1619 | 177 | 0.4072 | 0.5649 | 0.280 | 0.757 |
| najran | 1619 | 192 | 0.3630 | 0.5361 | 0.266 | 0.719 |
| qassim | 1619 | 122 | 0.3279 | 0.4950 | 0.214 | 0.738 |
| riyadh | 1619 | 141 | 0.3573 | 0.5299 | 0.237 | 0.766 |
| sharurah | 1619 | 105 | 0.2775 | 0.4733 | 0.215 | 0.676 |
| tabuk | 1619 | 133 | 0.3682 | 0.5117 | 0.236 | 0.722 |

## Figures

- `driver_ablation.png` — incremental PR-AUC by driver group (FDR)
- `pr_curve.png` — precision–recall curve for the forecast model
- `calibration.png` — reliability diagram (out-of-fold)
- `shap_importance.png` — SHAP feature importance

## Conclusion

The forecaster attains a cross-validated PR-AUC of 0.371 (ROC-AUC 0.841) at a 8.6% base rate, well above the no-skill PR-AUC of 0.086. After Benjamini-Hochberg FDR correction across the driver groups, **wind_direction** retains statistically significant incremental skill, with **wind_direction** the strongest (ΔPR-AUC +0.0523, FDR p=0.005). Remaining groups carry information already present elsewhere in the feature set.
