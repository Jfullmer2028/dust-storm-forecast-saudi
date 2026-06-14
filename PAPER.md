# Identifying the satellite and meteorological drivers of 24-hour dust-storm onset forecasting in Saudi Arabia

*A reproducible, keyless machine-learning pipeline for next-day dust-onset prediction.*

## Abstract

We forecast dust-storm onset 24 hours ahead at dust-prone Saudi Arabian stations
and identify **which** satellite and meteorological drivers carry predictive
skill. Using an XGBoost classifier with strict temporal cross-validation, we
train a forecaster on ERA5 reanalysis, MODIS vegetation and reflectivity, and
static soil properties, then run a systematic **driver-group ablation** that
quantifies each driver's *incremental* PR-AUC. On real 2018–2020 observations for
three stations (Riyadh, Hafar Al-Batin, Sharurah; 3,285 station-days, 6.1%
positive), the forecaster attains PR-AUC 0.14 / ROC-AUC 0.73, and the ablation
identifies **MODIS vegetation cover (NDVI) as the single driver group with
statistically significant incremental skill** (ΔPR-AUC +0.018, 95% CI
[+0.007, +0.037]). Where the satellite sees less green cover (a more exposed,
erodible surface), next-day dust is more predictable. The entire study runs
end-to-end with no API keys via public APIs (Open-Meteo, ORNL DAAC MODIS, NOAA
ISD, SoilGrids).

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

### 4.1 Forecast model performance (real data, 3 stations, 2018–2020)

Out-of-fold cross-validated skill of the forecaster:

| Metric | Value |
|--------|-------|
| PR-AUC (primary) | 0.14 |
| ROC-AUC | 0.73 |
| F₂ @ tuned threshold | 0.23 |

At a 6.1% base rate, ROC-AUC ≈ 0.73 indicates the forecaster has genuine
dust-ranking skill from the combined feature set.

### 4.2 Driver ablation

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
dust predictability beyond what the remaining features supply. The other groups —
including wind, which is informative but largely shared across features — do not
clear the significance threshold once the rest of the feature set is present.

### 4.3 Method validation on synthetic data

On a synthetic benchmark with two known satellite/surface driver signals — a
wind-direction (shamal) precursor and a vegetation precursor — the ablation
recovers **exactly those two groups** as the only significant drivers (ΔPR-AUC
+0.050 and +0.016), confirming the procedure isolates true drivers from noise.

![Driver ablation, synthetic](docs/assets/driver_ablation_synthetic.png)

## 5. Discussion

Vegetation cover emerging as the dominant satellite predictor is physically
coherent — NDVI indexes the fraction of bare, mobilisable surface, a first-order
control on dust emission that complements the wind and humidity fields rather than
duplicating them. The remaining driver groups carry information already present
elsewhere in the feature set, so they add no measurable skill once the others are
included. A ranked ablation of all driver groups, rather than a single
add-one-feature test, is what makes this separation visible.

## 6. Limitations

- Three stations and a single satellite-baseline year (2017); a ±20 km MODIS
  footprint rather than a basin-scale 200 km mean.
- Keyless satellite indices use MOD09A1 reflectance, not native higher-level
  MODIS products.
- Absolute skill is modest (PR-AUC ≈ 0.14 at a 6% base rate); dust onset at a
  point is intrinsically hard 24 h ahead.
- The synthetic results validate the *machinery*, not the geophysics.

These are single CLI flags away from being widened (`--albedo-km`,
`--modis-years`, `--stations`).

## 7. Conclusion

In a systematic ablation of satellite and reanalysis drivers, **MODIS vegetation
cover (NDVI) is the satellite variable that contributes statistically significant
incremental skill to 24-hour dust-storm forecasting** in Saudi Arabia, while the
other driver groups carry information already present elsewhere in the feature
set. The pipeline is fully reproducible without any API keys.

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
