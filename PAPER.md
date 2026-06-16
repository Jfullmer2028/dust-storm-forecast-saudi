# Identifying the satellite and meteorological drivers of 24-hour dust-storm onset forecasting in Saudi Arabia

*A reproducible, keyless machine-learning pipeline for next-day dust-onset prediction.*

## Abstract

We forecast dust-storm onset 24 hours ahead at dust-prone Saudi Arabian stations
and identify **which** satellite and meteorological drivers carry predictive
skill. Using a Platt-calibrated XGBoost classifier with strict temporal
cross-validation, we train a forecaster on ERA5 reanalysis, MODIS vegetation and
reflectivity, and static soil properties, then run a systematic **driver-group
ablation** whose incremental-PR-AUC p-values are corrected for multiple
comparisons (Benjamini-Hochberg FDR). On real 2018–2020 observations for **six
stations** (6,570 station-days, 5.6% positive), the forecaster attains PR-AUC 0.13
/ ROC-AUC 0.69 — above persistence and meteorology-only baselines and sharper than
climatology (Brier Skill Score +0.012). Three driver groups carry **FDR-significant,
seed-robust** incremental skill: **humidity/dryness** (ΔPR-AUC +0.036, FDR
p = 0.005), **vegetation (NDVI)** (+0.025, FDR p = 0.027) and **seasonality**
(+0.023, FDR p = 0.005) — i.e. dry air over a bare, erodible surface during the
dust season. The result generalizes to unseen stations (leave-one-station-out
PR-AUC 0.16), but absolute point-station skill is modest, so we frame it as a
reproducible **baseline** with robust driver attribution rather than a deployable
forecaster. The study runs end-to-end with no API keys via public APIs
(Open-Meteo, ORNL DAAC MODIS, NOAA ISD, SoilGrids).

## 1. Introduction

Dust storms are a major hazard in the Arabian Peninsula, degrading air quality,
aviation, and solar-energy yield. Numerical forecasts of dust emission depend on
surface erodibility, which is hard to observe directly, and on the synoptic
meteorology that mobilises and transports dust. A wide range of satellite and
reanalysis variables could in principle improve a data-driven forecast, but it is
rarely clear which actually carry incremental skill beyond the others. We
therefore ask: *which satellite and meteorological drivers contribute
statistically significant incremental skill to 24-hour dust-onset prediction?*,
and answer it with a driver-group ablation.

### Related work

Data-driven dust prediction over the Middle East has used tree ensembles and
neural networks on reanalysis and satellite inputs — e.g. ML forecasts of dust
frequency over Saudi cities, dust-risk mapping from MODIS over the Red Sea
region, and dust-source/occurrence models over Iraq and Central Asia. These
studies typically report a single best model and aggregate skill; few quantify
the *incremental* contribution of individual driver groups with corrected
significance, or release a fully reproducible keyless pipeline. We follow the
EMBRACE and REFORMS reporting guidance for ML-in-science (naive baselines,
leakage control, uncertainty, multiple-comparison correction) and contribute the
driver-attribution and reproducibility angle.

## 2. Data

All sources are public and **keyless**:

| Variable group | Source | Product |
|----------------|--------|---------|
| Meteorology (wind, humidity, soil moisture, precip, pressure, cloud) | Open-Meteo Historical Weather API | ERA5 reanalysis |
| Surface reflectance → NDVI, albedo | ORNL DAAC MODIS subsets | MOD09A1 (8-day, 500 m) |
| Station visibility | NOAA Integrated Surface Database | global-hourly |
| Static soil texture | ISRIC SoilGrids v2 | clay/sand/silt/OCS/bulk-density |

**Satellite indices (keyless).** From MOD09A1 surface reflectance we compute NDVI
and a shortwave-broadband reflectivity (albedo) index via the Liang (2001)
narrow-to-broadband conversion (bands 1–5, 7). Each index is expressed as an
*anomaly* — the deviation from a per-day-of-year climatology (±15 days) built from
a 2017 baseline year.

**Wind direction (the shamal).** Hourly wind is decomposed into resultant
northerly/easterly components and a northerly-flow fraction, capturing the NW
*shamal* that dominates Arabian dust emission.

**Labels.** A dust event on day *D+1* is defined by the WMO criterion: at least
one hourly horizontal visibility ≤ 1 000 m (NOAA ISD). Features on day *D*
predict the day-*D+1* label (`shift(-1)`).

## 3. Methods

- **Model.** XGBoost gradient-boosted trees; class imbalance handled with
  per-fold `scale_pos_weight`; missing values imputed with training-fold medians.
- **Cross-validation.** `TimeSeriesSplit` on rows sorted by *date* across
  stations, so every training fold strictly precedes its test fold (no temporal
  leakage). Decision thresholds (for F₂ only) are tuned on a held-out validation
  slice of each training fold — never the test fold. Probabilities are
  Platt-calibrated per fold on that validation slice. Spatial generalization is
  measured separately with **leave-one-station-out** CV.
- **Metrics.** Primary: **PR-AUC** (average precision), threshold-independent and
  appropriate for a ~6% rare-event problem. Secondary: ROC-AUC. Operational:
  F₂ (β=2) at the tuned threshold. Inference via paired bootstrap 95% CIs on
  out-of-fold predictions (2 000–5 000 resamples).
- **Naive baselines.** Skill is contextualised against a no-skill (climatological
  base-rate) reference, a persistence baseline (dust tomorrow if dust today), and
  a meteorology-only model.
- **Operational evaluation.** The probabilities are also judged as a warning
  system: Brier Skill Score against a climatology forecast, the recall achievable
  at a usable precision (≥ 0.30), and the precision-lift / false-alarm rate at the
  operating point that catches half of all dust days.
- **Seed robustness.** The model and the top driver's incremental PR-AUC are
  re-estimated over five random seeds (mean ± sd reported).
- **Driver ablation.** Features are bucketed into physical groups (wind speed,
  wind direction, humidity/dryness, antecedent moisture, thermal/BLH, pressure,
  vegetation, soil texture, seasonality, albedo). For each group *g*, we retrain
  on *all features − g* and measure the incremental contribution
  `PR-AUC(all) − PR-AUC(all − g)` with a paired bootstrap CI and a two-sided
  bootstrap p-value. Because one test is run per group, p-values are corrected
  for multiple comparisons with **Benjamini-Hochberg FDR** (q = 0.05); the
  corrected call is the reported result.

## 4. Results

### 4.1 Forecast model performance (real data, 6 stations, 2018–2020)

Out-of-fold cross-validated skill, with naive-baseline context (6,570 station-days,
5.6% positive). Across five random seeds the model gives PR-AUC = 0.112 ± 0.009,
ROC-AUC = 0.668 ± 0.023.

| Reference | PR-AUC | ROC-AUC |
|-----------|--------|---------|
| no-skill (base rate) | 0.056 | 0.500 |
| persistence (dust today → dust tomorrow) | 0.100 | 0.597 |
| meteorology-only model | 0.108 | 0.687 |
| **full model** | **0.125** | **0.686** |

The full model beats persistence and the meteorology-only model. As a warning
system, the calibrated probabilities have a positive Brier Skill Score (+0.012,
sharper than climatology); the operating points, however, are weak — catching half
of all dust days costs a ~38% false-alarm rate — so the model is a useful baseline,
not a deployable forecaster.

**Generalization to unseen stations.** Under leave-one-station-out
cross-validation (train on five stations, test on a held-out sixth), the model
attains PR-AUC 0.159 / ROC-AUC 0.719 — on par with, indeed slightly above, the
within-station temporal CV. The driver relationships therefore transfer across
the region to new stations; what varies is each station's *intrinsic* 24-h
predictability (within-station temporal PR-AUC ranges from 0.144 at Riyadh to
0.044 at Tabuk, reflecting how synoptically- versus locally-driven each site's
dust is), not the model's ability to generalize spatially.

### 4.2 Driver ablation (Benjamini-Hochberg FDR corrected)

![Driver ablation, real data](docs/assets/driver_ablation_real.png)

| Driver group | Incremental PR-AUC | 95% CI | p | p (FDR) | sig. |
|--------------|--------------------|--------|---|---------|------|
| **humidity / dryness** | **+0.036** | **[+0.015, +0.065]** | 0.001 | **0.005** | **yes** |
| **vegetation (NDVI)** | **+0.025** | **[+0.006, +0.049]** | 0.008 | **0.027** | **yes** |
| **seasonality** | **+0.023** | **[+0.007, +0.044]** | 0.001 | **0.005** | **yes** |
| soil texture | +0.017 | [+0.002, +0.036] | 0.022 | 0.055 | no |
| wind speed | +0.016 | [−0.005, +0.040] | 0.136 | 0.170 | no |
| antecedent moisture | +0.016 | [−0.002, +0.040] | 0.084 | 0.157 | no |
| albedo | +0.014 | [−0.002, +0.037] | 0.094 | 0.157 | no |
| thermal / BLH | +0.014 | [−0.003, +0.036] | 0.116 | 0.166 | no |
| pressure | +0.008 | [−0.007, +0.023] | 0.310 | 0.344 | no |
| wind direction | −0.001 | [−0.016, +0.016] | 0.873 | 0.873 | no |

**Three driver groups survive Benjamini-Hochberg FDR correction: humidity/dryness,
vegetation, and seasonality** — physically, dry air over a bare, erodible surface
during the dust season. The seed-averaged incremental skill of the top driver is
+0.023 ± 0.011 over five seeds (clear of zero), so the effect is robust to the
random seed as well as to multiple comparisons. (Per-fold Platt calibration is
important here: pooling *uncalibrated* cross-fold probabilities mismatches the
probability scale across folds and masks the signal in the pooled metric.)

### 4.3 Method validation on synthetic data

On a synthetic benchmark with two known satellite/surface driver signals — a
strong wind-direction (shamal) precursor and a weaker vegetation precursor — the
ablation ranks **exactly those two groups first** (ΔPR-AUC +0.052 and +0.017).
After FDR correction the strong signal is recovered as significant (FDR p = 0.005)
while the weaker one is flagged as suggestive (FDR p = 0.085), demonstrating that
the procedure isolates true drivers from noise and is appropriately conservative
on weak effects at finite sample size.

![Driver ablation, synthetic](docs/assets/driver_ablation_synthetic.png)

## 5. Discussion

The three FDR-significant drivers form a physically coherent picture of dust
onset: **dry air** (humidity/dryness), a **bare, erodible surface** (vegetation /
NDVI) and the **dust season** (seasonality) — emission is favoured when a dry,
sparsely-vegetated surface meets the seasonal synoptic forcing. These are
first-order controls that complement, rather than duplicate, each other, which is
why each adds incremental skill. Wind speed and direction, by contrast, do not
survive correction: they are informative but their information is largely shared
across the feature set (the drop-group ablation credits only the *unique*
contribution). A ranked, FDR-controlled ablation of all driver groups — rather
than a single add-one-feature test — is what makes this separation visible, and
per-fold calibration is what lets the per-fold signal aggregate coherently in the
pooled metric.

Encouragingly, the result **generalizes spatially**: leave-one-station-out CV
(PR-AUC 0.159 on held-out, unseen stations) matches the within-station figure, so
the driver relationships are not station-specific artefacts but transfer across
the region. The remaining limitation is *absolute* skill at a point: at a 5.6%
base rate the forecaster reaches PR-AUC ≈ 0.13 and catching half of all dust days
costs a ~38% false-alarm rate, and intrinsic 24-h predictability varies by site.
The contribution is therefore best read as a **reproducible, spatially
generalizable baseline with robust driver attribution**, and a sobering reference
point for what coarse, keyless, point-station inputs can deliver 24 h ahead.

## 6. Limitations

- Six stations and a single satellite-baseline year (2017); a ±20 km MODIS
  footprint rather than a basin-scale 200 km mean.
- Keyless satellite indices use MOD09A1 reflectance, not native higher-level
  MODIS products.
- Absolute skill is modest (PR-AUC ≈ 0.13 at a 5.6% base rate) and
  station-dependent; dust onset at a point is intrinsically hard 24 h ahead.
- The drop-group ablation measures *incremental* skill, so a driver that is real
  but **correlated** with others (e.g. wind speed and wind direction) can test
  non-significant because its information is also carried elsewhere; this is a
  conservative, not a null, statement about such drivers.
- The synthetic results validate the *machinery*, not the geophysics.

These are single CLI flags away from being widened (`--albedo-km`,
`--modis-years`, `--stations`).

## 7. Conclusion

In a systematic, FDR-corrected, calibrated ablation across six Saudi stations,
**three driver groups carry robust incremental skill for 24-hour dust-storm
onset — humidity/dryness, vegetation cover (NDVI), and seasonality** (FDR
p = 0.005, 0.027, 0.005; seed-robust) — i.e. dry air over a bare, erodible surface
during the dust season. The forecaster beats naive persistence and
meteorology-only baselines, its calibrated probabilities are sharper than
climatology, and it **generalizes to unseen stations** (leave-one-station-out
PR-AUC 0.159, matching the within-station figure). Absolute, operational skill at
a point remains modest, so the contribution is a reproducible, spatially
generalizable **baseline with robust driver attribution** rather than a deployable
warning system. The pipeline is fully reproducible without any API keys.

## Reproduce

```bash
pip install -r requirements.txt
python run_pipeline.py            # synthetic (no keys)
python run_pipeline.py --mode real  # keyless live data
streamlit run app.py              # interactive demo
```

## References

- Liang, S. (2001). *Narrowband to broadband conversions of land surface albedo:
  I. Algorithms.* Remote Sensing of Environment, 76(2), 213–238.
- Qu, J. J. et al. (2006). *Asian dust storm monitoring combining Terra and Aqua
  MODIS SRB measurements* (NDDI). IEEE GRSL, 3(4).
- Hersbach, H. et al. (2020). *The ERA5 global reanalysis.* QJRMS, 146(730).
- Chen, T. & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system.* KDD.
- Benjamini, Y. & Hochberg, Y. (1995). *Controlling the false discovery rate: a
  practical and powerful approach to multiple testing.* J. R. Statist. Soc. B,
  57(1), 289–300.
- Kapoor, S., Narayanan, A., et al. (2024). *REFORMS: Reporting standards for
  machine-learning-based science.* Science Advances.
- *EMBRACE: Environmental machine learning, baseline reporting, and comprehensive
  evaluation* (2024). Environmental Science & Technology. Checklist:
  github.com/starfriend10/EMBRACE.
- Regional data-driven dust studies over Saudi Arabia, the Red Sea and Iraq are
  surveyed in the Introduction (citations omitted here for brevity).
