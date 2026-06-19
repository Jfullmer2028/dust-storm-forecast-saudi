# Identifying the satellite and meteorological drivers of 24-hour dust-storm onset forecasting in Saudi Arabia

*A reproducible, keyless machine-learning pipeline for next-day dust-onset prediction.*

## Abstract

We forecast dust-storm onset 24 hours ahead at dust-prone Saudi Arabian stations
and ask **which** satellite and meteorological drivers carry genuine incremental
predictive skill. Using a Platt-calibrated XGBoost classifier with strict
temporal cross-validation, we train a forecaster on ERA5 reanalysis, MODIS
vegetation and reflectivity, and static soil properties, then run a systematic
**driver-group ablation** whose incremental-PR-AUC p-values are corrected for
multiple comparisons (Benjamini-Hochberg FDR) and then stress-tested for
robustness across random seeds and across stations. On real 2018–2020
observations for **six stations** (6,570 station-days, 5.6% positive), the
forecaster attains PR-AUC 0.125 / ROC-AUC 0.686 — above persistence and
meteorology-only baselines, well above a day-of-year seasonal climatology, and
sharper than climatology as a probability forecast (Brier Skill Score +0.012,
Expected Calibration Error 0.005). Three driver groups survive FDR correction —
**humidity/dryness** (ΔPR-AUC +0.036, FDR p = 0.005), **vegetation (NDVI)**
(+0.025, FDR p = 0.027) and **seasonality** (+0.023, FDR p = 0.005) — but only
**humidity/dryness** also survives the stricter robustness bar (stable across
seeds *and* sign-consistent under station jackknife). We therefore report
humidity/dryness as the one **fully robust** driver and vegetation and
seasonality as FDR-significant but **suggestive**. The result generalizes to
unseen stations (leave-one-station-out PR-AUC 0.159, on par with within-station),
but absolute point-station skill is modest, so we frame it as a reproducible
**baseline with one robust, physically coherent driver attribution** rather than
a deployable forecaster. The study runs end-to-end with no API keys via public
APIs (Open-Meteo, ORNL DAAC MODIS, NOAA ISD, SoilGrids).

## 1. Introduction

Dust storms are a major hazard in the Arabian Peninsula, degrading air quality,
aviation, and solar-energy yield. Numerical forecasts of dust emission depend on
surface erodibility, which is hard to observe directly, and on the synoptic
meteorology that mobilises and transports dust. A wide range of satellite and
reanalysis variables could in principle improve a data-driven forecast, but it is
rarely clear which actually carry incremental skill beyond the others, and rarer
still whether a claimed driver survives the random choices (seed, station split)
that any single fit conceals. We therefore ask: *which satellite and
meteorological drivers contribute statistically significant **and robust**
incremental skill to 24-hour dust-onset prediction?*, and answer it with a
driver-group ablation that is FDR-corrected and then re-tested across seeds and
across stations.

### Related work

Data-driven dust prediction over the Middle East has used tree ensembles and
neural networks on reanalysis and satellite inputs — e.g. ML forecasts of dust
frequency over Saudi cities, dust-risk mapping from MODIS over the Red Sea
region, and dust-source/occurrence models over Iraq and Central Asia. These
studies typically report a single best model and aggregate skill; few quantify
the *incremental* contribution of individual driver groups with corrected
significance, fewer still subject those contributions to seed/station robustness
checks, and few release a fully reproducible keyless pipeline. We follow the
EMBRACE and REFORMS reporting guidance for ML-in-science (naive baselines,
leakage control, uncertainty, multiple-comparison correction) and contribute the
robust driver-attribution and reproducibility angle.

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
predict the day-*D+1* label.

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
  base-rate) reference, a persistence baseline (dust tomorrow if dust today), a
  **day-of-year seasonal climatology** (out-of-fold smoothed dust-rate look-up —
  the right bar for a "seasonality" claim), and a meteorology-only model.
- **Operational evaluation.** The probabilities are also judged as a warning
  system: Brier Skill Score and **Expected Calibration Error** (10-bin) against a
  climatology forecast, the recall achievable at a usable precision (≥ 0.30), and
  the precision-lift / false-alarm rate at the operating point that catches half
  of all dust days.
- **Driver ablation.** Features are bucketed into physical groups (wind speed,
  wind direction, humidity/dryness, antecedent moisture, thermal/BLH, pressure,
  vegetation, soil texture, seasonality, albedo). For each group *g*, we retrain
  on *all features − g* and measure the incremental contribution
  `PR-AUC(all) − PR-AUC(all − g)` with a paired bootstrap CI and a two-sided
  bootstrap p-value. Because one test is run per group, p-values are corrected
  for multiple comparisons with **Benjamini-Hochberg FDR** (q = 0.05).
- **Robustness of the ablation (the key tightening).** FDR significance from a
  single fit is necessary but not sufficient. For every FDR-significant driver we
  additionally require (i) **seed robustness** — its incremental PR-AUC stays
  above zero by at least one standard deviation when re-estimated over five
  random training seeds (mean − sd > 0); and (ii) **station-jackknife
  sign-consistency** — its incremental PR-AUC stays positive on ≥ 80% of
  leave-one-station-out subsets. A driver is called **fully robust** only if it
  passes FDR *and* both checks; one that passes FDR alone is reported as
  *suggestive*.

## 4. Results

### 4.1 Forecast model performance (real data, 6 stations, 2018–2020)

Out-of-fold cross-validated skill, with naive-baseline context (6,570 station-days,
5.6% positive). Across five random seeds the model gives PR-AUC = 0.112 ± 0.009,
ROC-AUC = 0.668 ± 0.023.

| Reference | PR-AUC | ROC-AUC |
|-----------|--------|---------|
| no-skill (base rate) | 0.056 | 0.500 |
| persistence (dust today → dust tomorrow) | 0.100 | 0.597 |
| seasonal climatology (day-of-year) | 0.061 | 0.607 |
| meteorology-only model | 0.108 | 0.687 |
| **full model** | **0.125** | **0.686** |

The full model beats persistence, the day-of-year seasonal climatology, and the
meteorology-only model. The day-of-year climatology is barely above no-skill
(PR-AUC 0.061 vs 0.056), so *when* in the year dust occurs carries very little
standalone predictive content here — a fact that bears directly on the
seasonality driver below. As a warning system, the calibrated probabilities are
sharp and well-calibrated (Brier Skill Score +0.012, Expected Calibration Error
0.005), but the operating points are weak — catching half of all dust days costs
a ~38% false-alarm rate at 6% precision — so the model is a useful baseline, not
a deployable forecaster.

**Generalization to unseen stations.** Under leave-one-station-out
cross-validation (train on five stations, test on a held-out sixth), the model
attains PR-AUC 0.159 / ROC-AUC 0.719 — on par with, indeed slightly above, the
within-station temporal CV. The driver relationships therefore transfer across
the region to new stations; what varies is each station's *intrinsic* 24-h
predictability (within-station PR-AUC ranges from 0.144 at Riyadh to 0.044 at
Tabuk), not the model's ability to generalize spatially.

### 4.2 Driver ablation (FDR-corrected, then robustness-tested)

![Driver ablation, real data](docs/assets/driver_ablation_real.png)

| Driver group | Incremental PR-AUC | 95% CI | p | p (FDR) | FDR sig. |
|--------------|--------------------|--------|---|---------|----------|
| **humidity / dryness** | **+0.036** | **[+0.015, +0.065]** | 0.001 | **0.005** | **yes** |
| **vegetation (NDVI)** | **+0.025** | **[+0.006, +0.049]** | 0.008 | **0.027** | **yes** |
| **seasonality** | **+0.023** | **[+0.007, +0.044]** | 0.001 | **0.005** | **yes** |
| soil texture | +0.017 | [+0.002, +0.036] | 0.022 | 0.055 | no |
| wind speed | +0.016 | [−0.005, +0.040] | 0.136 | 0.170 | no |
| antecedent moisture | +0.016 | [−0.002, +0.040] | 0.084 | 0.157 | no |
| albedo | +0.014 | [−0.002, +0.037] | 0.094 | 0.157 | no |
| thermal / BLH | +0.014 | [−0.003, +0.037] | 0.116 | 0.166 | no |
| pressure | +0.008 | [−0.007, +0.023] | 0.310 | 0.344 | no |
| wind direction | −0.001 | [−0.016, +0.016] | 0.873 | 0.873 | no |

Three driver groups survive Benjamini-Hochberg FDR correction: humidity/dryness,
vegetation, and seasonality. We then subject each to the seed and station
robustness checks:

| FDR-significant driver | Seed ΔPR-AUC (mean ± sd, min) | Seed robust? | Station-jackknife positive | Verdict |
|------------------------|------------------------------|--------------|----------------------------|---------|
| **humidity / dryness** | +0.023 ± 0.011 (min +0.003) | **yes** | **6 / 6** | **fully robust** |
| vegetation (NDVI) | +0.005 ± 0.008 (min −0.006) | no | 3 / 6 | suggestive |
| seasonality | +0.008 ± 0.012 (min −0.009) | no | 3 / 6 | suggestive |

**Only humidity/dryness is fully robust.** Its incremental skill is positive on
*every* seed (mean − sd > 0) and on *every* leave-one-station-out subset.
Vegetation and seasonality, although FDR-significant in the single full-sample
fit, are fragile: their seed-averaged incremental skill is within one standard
deviation of zero, and each adds skill on only half of the station-jackknife
subsets. We therefore report them as **suggestive, not established**. For
seasonality this is consistent with §4.1: a standalone day-of-year climatology is
barely better than no-skill, so it is unsurprising that seasonality's incremental
contribution does not hold up across resampling. (Per-fold Platt calibration
matters here: pooling *uncalibrated* cross-fold probabilities mismatches the
probability scale across folds and masks the signal in the pooled metric; with
calibration the well-calibrated ECE of 0.005 confirms the probabilities are
honest.)

### 4.3 Method validation on synthetic data

On a synthetic benchmark with two known satellite/surface driver signals — a
strong wind-direction (shamal) precursor and a weaker vegetation precursor — the
ablation ranks **exactly those two groups first** (ΔPR-AUC +0.052 and +0.017).
After FDR correction the strong signal is recovered as significant (FDR p = 0.005)
*and* fully robust (positive on every seed and on all 8 station-jackknife
subsets), while the weaker one is flagged as suggestive (FDR p = 0.085). This
demonstrates that the procedure isolates true drivers from noise, recovers a
genuine strong driver as robust, and is appropriately conservative on weak
effects at finite sample size.

![Driver ablation, synthetic](docs/assets/driver_ablation_synthetic.png)

## 5. Discussion

After the stricter bar, the defensible attribution narrows to a single,
physically first-order control: **dry air** (humidity/dryness). It is the one
driver whose incremental skill survives FDR correction *and* re-estimation across
seeds *and* removal of any single station — i.e. it is neither a multiple-testing
artefact nor an artefact of one station or one lucky fit. Physically this is the
expected leading control: a dry boundary layer favours dust lofting and reduces
the cohesion that suppresses emission.

Vegetation (NDVI) and seasonality remain *suggestive*. Their point estimates are
positive and FDR-significant in the full sample, and they fit the same physical
story (a bare, erodible surface during the dust season), but at this sample size
their contribution is not stable enough to call established. Honest reporting
requires distinguishing the two tiers rather than presenting all three FDR hits
as equally secure. The seasonal-climatology baseline reinforces this for
seasonality specifically: a calendar look-up alone barely beats no-skill, so the
seasonal "driver" is doing little that the rest of the feature set does not
already capture.

Encouragingly, the result **generalizes spatially**: leave-one-station-out CV
(PR-AUC 0.159 on held-out, unseen stations) matches the within-station figure, so
the driver relationships are not station-specific artefacts but transfer across
the region. The remaining limitation is *absolute* skill at a point: at a 5.6%
base rate the forecaster reaches PR-AUC ≈ 0.13 and catching half of all dust days
costs a ~38% false-alarm rate, and intrinsic 24-h predictability varies by site.
The contribution is therefore best read as a **reproducible, spatially
generalizable baseline with one robust driver attribution (humidity/dryness)**,
and a sobering reference point for what coarse, keyless, point-station inputs can
deliver 24 h ahead.

## 6. Limitations

- Six stations and a single satellite-baseline year (2017); a ±20 km MODIS
  footprint rather than a basin-scale 200 km mean. Widening these (`--stations`,
  `--modis-years`, `--albedo-km`) is the natural route to upgrade vegetation and
  seasonality from suggestive to robust — or to rule them out.
- Keyless satellite indices use MOD09A1 reflectance, not native higher-level
  MODIS products.
- Absolute skill is modest (PR-AUC ≈ 0.13 at a 5.6% base rate); dust onset at a
  point is intrinsically hard 24 h ahead.
- The drop-group ablation measures *incremental* skill, so a driver that is real
  but **correlated** with others (e.g. wind speed and wind direction) can test
  non-significant because its information is also carried elsewhere; this is a
  conservative, not a null, statement about such drivers.
- The robustness bar is deliberately strict (seed mean − sd > 0 *and* ≥ 80%
  station-jackknife positivity); a driver labelled "suggestive" here is not
  refuted, only not yet established.
- The synthetic results validate the *machinery*, not the geophysics.

## 7. Conclusion

In a systematic, FDR-corrected, calibrated ablation across six Saudi stations —
then stress-tested across seeds and stations — **one driver group carries fully
robust incremental skill for 24-hour dust-storm onset: humidity/dryness** (FDR
p = 0.005; positive on every seed and every station-jackknife subset). Two further
groups, **vegetation (NDVI)** and **seasonality**, are FDR-significant but do not
survive the seed/station robustness checks, so we report them as suggestive
rather than established. The forecaster beats naive persistence, seasonal
climatology, and meteorology-only baselines, its calibrated probabilities are
sharp and well-calibrated (BSS +0.012, ECE 0.005), and it **generalizes to unseen
stations** (leave-one-station-out PR-AUC 0.159). Absolute, operational skill at a
point remains modest, so the contribution is a reproducible, spatially
generalizable **baseline with one robust driver attribution**, not a deployable
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
