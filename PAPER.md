# Which satellite drivers improve 24-hour dust-storm forecasting in Saudi Arabia? A driver-ablation study

*A reproducible, keyless machine-learning pipeline for next-day dust-onset prediction.*

## Abstract

We test whether satellite-derived surface information improves 24-hour-ahead
prediction of dust-storm onset at dust-prone Saudi Arabian stations, and — more
usefully — we identify **which** satellite/surface drivers actually matter. Using
an XGBoost classifier with strict temporal cross-validation, we compare a
meteorological baseline (ERA5 reanalysis + static soil + NDVI) against models
that add MODIS shortwave-broadband **albedo** anomalies, and we run a systematic
**driver ablation** that quantifies each physical driver group's *incremental*
PR-AUC. On real 2018–2020 observations for three stations (Riyadh, Hafar
Al-Batin, Sharurah; 3,285 station-days, 6.1% positive), **adding satellite albedo
does not significantly improve forecasting** (ΔPR-AUC +0.004, 95% CI
[−0.015, +0.022]). However, the ablation reveals that **MODIS vegetation cover
(NDVI) is the single driver group with significant incremental skill**
(ΔPR-AUC +0.018, 95% CI [+0.007, +0.037]). The operative answer is therefore not
albedo but **surface vegetation state**. The entire study runs end-to-end with no
API keys via public APIs (Open-Meteo, ORNL DAAC MODIS, NOAA ISD, SoilGrids).

## 1. Introduction

Dust storms are a major hazard in the Arabian Peninsula, degrading air quality,
aviation, and solar-energy yield. Numerical forecasts of dust emission depend on
surface erodibility, which is hard to observe directly. Satellite surface
reflectivity (albedo) has been proposed as a proxy for erodible, bright, exposed
surfaces. We ask: *does adding satellite albedo improve 24-hour dust-onset
prediction over a competent meteorological baseline?* Recognising that a single
binary test yields limited insight, we generalise the question to: *which
satellite and surface drivers carry incremental predictive skill?*

## 2. Data

All sources are public and **keyless**:

| Variable group | Source | Product |
|----------------|--------|---------|
| Meteorology (wind, humidity, soil moisture, precip, pressure, cloud) | Open-Meteo Historical Weather API | ERA5 reanalysis |
| Surface reflectance → NDVI, albedo | ORNL DAAC MODIS subsets | MOD09A1 (8-day, 500 m) |
| Station visibility | NOAA Integrated Surface Database | global-hourly |
| Static soil texture | ISRIC SoilGrids v2 | clay/sand/silt/OCS/bulk-density |

**Albedo (keyless).** ORNL does not serve MCD43A3 albedo globally, so we derive a
shortwave-broadband albedo from MOD09A1 surface reflectance via the Liang (2001)
narrow-to-broadband conversion (bands 1–5, 7). The albedo *anomaly* is the
deviation from a per-day-of-year climatology (±15 days) built from a 2017
baseline year.

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
  slice of each training fold — never the test fold.
- **Metrics.** Primary: **PR-AUC** (average precision), threshold-independent and
  appropriate for a ~6% rare-event problem. Secondary: ROC-AUC. Operational:
  F₂ (β=2) at the tuned threshold. Inference via paired bootstrap 95% CIs on
  out-of-fold predictions (5 000 resamples) and Wilcoxon signed-rank on per-fold
  scores.
- **Driver ablation.** Features are bucketed into physical groups (wind speed,
  wind direction, humidity/dryness, antecedent moisture, thermal/BLH, pressure,
  vegetation, soil texture, seasonality, albedo). For each group *g*, we retrain
  on *all features − g* and measure the incremental contribution
  `PR-AUC(all) − PR-AUC(all − g)` with a paired bootstrap CI. A CI entirely above
  zero marks a driver that supplies skill no other group does.

## 4. Results

### 4.1 Albedo head-to-head (real data, 3 stations, 2018–2020)

| Metric | Baseline | + Albedo | Δ | 95% CI |
|--------|----------|----------|---|--------|
| PR-AUC (primary) | 0.117 | 0.116 | −0.002 | [−0.021, +0.018] |
| ROC-AUC | 0.713 | 0.718 | +0.005 | [−0.015, +0.025] |
| F₂ @ tuned thr | 0.265 | 0.257 | −0.008 | [−0.035, +0.053] |

All three metrics agree: adding satellite albedo does not significantly change
performance. ROC-AUC ≈ 0.71 confirms both models have genuine skill — the
meteorological baseline already captures the predictable signal.

### 4.2 Driver ablation — what actually matters

![Driver ablation, real data](docs/assets/driver_ablation_real.png)

| Driver group | Incremental PR-AUC | 95% CI | Significant |
|--------------|--------------------|--------|-------------|
| **vegetation (NDVI)** | **+0.018** | **[+0.007, +0.037]** | **yes** |
| antecedent moisture | +0.008 | [−0.004, +0.021] | no |
| seasonality | +0.008 | [−0.004, +0.021] | no |
| pressure | +0.007 | [−0.002, +0.018] | no |
| wind direction | +0.005 | [−0.018, +0.021] | no |
| wind speed | +0.005 | [−0.023, +0.027] | no |
| albedo | +0.004 | [−0.015, +0.022] | no |
| humidity / dryness | −0.002 | [−0.018, +0.012] | no |
| soil texture | −0.003 | [−0.013, +0.007] | no |

**Vegetation (NDVI) is the only driver group with a significant incremental
contribution.** Lower green cover (more exposed, erodible surface) raises next-day
dust predictability beyond what meteorology supplies.

### 4.3 Method validation on synthetic data

On a synthetic benchmark with two known driver signals — a wind-direction (shamal)
precursor and an albedo-erodibility precursor — the ablation recovers **exactly
those two groups** as the only significant drivers (ΔPR-AUC +0.070 and +0.061),
confirming the procedure isolates true drivers from noise.

![Driver ablation, synthetic](docs/assets/driver_ablation_synthetic.png)

## 5. Discussion

Vegetation cover emerging as the dominant satellite predictor is physically
coherent — NDVI indexes the fraction of bare, mobilisable surface, a first-order
control on dust emission that complements (rather than duplicates) the wind and
humidity fields. Albedo, by contrast, adds no measurable skill beyond NDVI and
meteorology, plausibly because over uniformly bright desert the broadband-albedo
anomaly is weak and noisy relative to vegetation contrast. Generalising the study
from a single albedo test to a ranked ablation of all driver groups is what
surfaces this result.

## 6. Limitations

- Three stations and a single albedo-baseline year (2017); a ±20 km MODIS
  footprint rather than a basin-scale 200 km mean.
- Keyless albedo is a Liang reflectance proxy, not native MCD43A3.
- Absolute skill is modest (PR-AUC ≈ 0.12 at a 6% base rate); dust onset at a
  point is intrinsically hard 24 h ahead.
- The synthetic results validate the *machinery*, not the geophysics.

These are single CLI flags away from being widened (`--albedo-km`,
`--modis-years`, `--stations`).

## 7. Conclusion

In a systematic ablation of satellite and reanalysis drivers, **MODIS vegetation
cover (NDVI) — not albedo — is the satellite variable that significantly improves
24-hour dust-storm forecasting** in Saudi Arabia. The pipeline is fully
reproducible without any API keys.

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
