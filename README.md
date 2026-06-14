# Dust-Storm Forecast — Saudi Arabia

Forecast dust-storm onset **24 hours in advance** at Saudi Arabian weather stations using XGBoost, and **identify which satellite and surface drivers carry forecasting skill**. The pipeline trains a forecaster on ERA5 meteorology, MODIS vegetation and reflectivity, and static soil properties, then runs a **driver-group ablation** that measures each physical driver's *incremental* contribution to forecast skill.

> **Headline finding (real data, Riyadh/Hafar/Sharurah, 2018–2020):** **MODIS vegetation cover (NDVI) is the only driver group whose incremental skill survives multiple-comparison (Benjamini–Hochberg FDR) correction** (ΔPR-AUC **+0.018, 95% CI [+0.007, +0.037], FDR p = 0.030**), above naive persistence and meteorology-only baselines. Where the satellite sees less green cover (more exposed, erodible surface), next-day dust is more predictable. See [`results/report_real.md`](results/report_real.md).

| Station | Region | WMO ID | Coordinates |
|---------|--------|--------|-------------|
| Riyadh (King Khalid Intl.) | Central | 404380 | 24.93°N, 46.72°E |
| Hafar Al-Batin | North-east | 403730 | 28.33°N, 46.13°E |
| Sharurah | South | 411400 | 17.47°N, 47.12°E |
| Dammam (Dhahran) | East | 404160 | 26.27°N, 50.15°E |
| Tabuk | North-west | 403750 | 28.37°N, 36.60°E |
| Qassim (Buraidah) | Central-north | 404050 | 26.30°N, 43.77°E |
| Arar | North | 403570 | 30.91°N, 41.14°E |
| Najran | South-west | 411370 | 17.61°N, 44.42°E |

> WMO identifiers are best-effort for the real-data path; **coordinates are authoritative** and drive every geospatial extraction.

## Quick Start (Synthetic Data — No API Keys)

```bash
git clone https://github.com/jfullmer2028/dust-storm-forecast-saudi.git
cd dust-storm-forecast-saudi
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py
```

The pipeline will:
1. Load (or generate) synthetic test data in `data/synthetic/`
2. Engineer features: ERA5 daily aggregates (incl. wind-direction components), MODIS NDVI and shortwave reflectivity, static soil properties, and cyclical temporal encodings
3. Cross-validate the forecaster with `TimeSeriesSplit` (8 folds), tuning an **F₂-optimal decision threshold** on a held-out validation slice of each fold
4. Run a **driver-group ablation** — drop each physical driver group, retrain, and measure its incremental **PR-AUC** with a paired **bootstrap 95% CI**
5. Report per-fold and per-station performance, and write figures (ablation chart, PR curve, SHAP) to `outputs/` and `results/report.md`

Expected runtime: ~6–10 minutes on a laptop (~14.5k station-days; the ablation retrains per driver group).

### Metrics — why PR-AUC is primary

F₂ depends on a decision threshold tuned per fold; with dust events at ~6 % those
validation slices are small, so the threshold is sensitive to noise. The
**primary metric is therefore PR-AUC (average precision)** — threshold-independent
and the field standard for rare-event detection — computed on the out-of-fold
predicted probabilities. **ROC-AUC** is reported alongside, and **F₂ at the tuned
threshold** is kept as the *operational* metric.

### Representative result (synthetic mode, 8-fold CV)

The synthetic generator builds in two known satellite/surface drivers (a strong
wind-direction precursor and a weaker vegetation precursor) as a benchmark on
which the ablation can be checked. The ablation ranks **exactly those two groups
first** (ΔPR-AUC +0.050 and +0.016); after FDR correction the strong signal is
recovered as significant (FDR p = 0.005) and the weak one is flagged as suggestive
(FDR p = 0.070) — the procedure isolates true drivers and stays conservative on
weak effects.

### Real-data result (keyless `--mode real`, Riyadh/Hafar/Sharurah, 2018–2020)

Run end-to-end on **live observations** (Open-Meteo ERA5 + ORNL MODIS + NOAA ISD + SoilGrids): 3,285 station-days, 201 dust events (6.1 %). Across 5 seeds the forecaster reaches **PR-AUC 0.130 ± 0.006 / ROC-AUC 0.721 ± 0.006**, above naive baselines:

| Reference | PR-AUC | ROC-AUC |
|-----------|--------|---------|
| no-skill (base rate) | 0.061 | 0.500 |
| persistence | 0.102 | 0.592 |
| meteorology-only model | 0.119 | 0.711 |
| **full model** | **0.141** | **0.730** |

The driver ablation, ranked by incremental PR-AUC, with **Benjamini–Hochberg FDR-corrected** p-values:

| Driver group | Incremental PR-AUC | 95% CI | p (FDR) | sig. |
|--------------|--------------------|--------|---------|------|
| **vegetation (NDVI)** | **+0.018** | **[+0.007, +0.037]** | **0.030** | ✅ **yes** |
| antecedent moisture | +0.008 | [−0.004, +0.021] | 0.438 | no |
| seasonality | +0.008 | [−0.004, +0.021] | 0.438 | no |
| pressure | +0.007 | [−0.002, +0.018] | — | no |
| wind direction | +0.005 | [−0.018, +0.021] | — | no |
| wind speed | +0.005 | [−0.023, +0.027] | — | no |
| albedo | +0.004 | [−0.015, +0.022] | — | no |
| humidity/dryness | −0.002 | [−0.018, +0.012] | — | no |
| soil texture | −0.003 | [−0.013, +0.007] | — | no |

**Vegetation cover (NDVI) is the only group whose incremental skill survives FDR correction** (FDR p = 0.030): where the satellite sees less green cover (more exposed, erodible surface), next-day dust is more predictable. As an honesty check, the top driver's seed-averaged Δ is +0.009 ± 0.008 (seed-sensitive at this sample). This uses 3 stations and a ±20 km MODIS footprint; widening scope (`--stations`, `--modis-years`) is the natural confirmation step.

## Project Structure

```
dust-storm-forecast-saudi/
├── config.yaml              # Stations, paths, model + real-mode settings
├── run_pipeline.py          # Master script (synthetic / keyless real)
├── app.py                   # 🎛️ Streamlit interactive demo
├── PAPER.md                 # 📄 Succinct paper (abstract → conclusion)
├── docs/index.html          # 🌐 Live GitHub Pages project page + assets/
├── requirements.txt         # Core deps · requirements-real.txt · requirements-demo.txt
├── .github/workflows/       # ci.yml (tests + pipeline) · pages.yml (Pages deploy)
├── data/
│   ├── synthetic/           # Pre-built test CSVs (no API keys needed)
│   ├── raw/real/            # Cached keyless real downloads (gitignored)
│   └── final/               # master_dataset[_real].csv (generated)
├── src/
│   ├── acquisition.py       # GEE/CDS/ISD acquisition + synthetic generator
│   ├── real_sources.py      # Keyless real APIs (Open-Meteo/ORNL/NOAA/SoilGrids)
│   ├── features.py          # Albedo anomaly, lags, encodings, feature sets, driver groups
│   ├── labeling.py          # Dual-criterion & visibility-only labels
│   ├── models.py            # XGBoost CV, PR-AUC/ROC-AUC/F₂, threshold tuning, Optuna
│   └── evaluation.py        # Metric comparison, driver ablation, bootstrap, per-station, SHAP
├── tests/                   # Pytest suite (features, labels, CV, real sources, ablation, demo)
├── outputs/                 # Figures incl. driver_ablation.png, pr_curves.png (outputs/real/ too)
└── results/
    ├── report.md            # Synthetic-mode results (generated)
    └── report_real.md       # Keyless real-data results (generated)
```

## Tests

```bash
pip install pytest
pytest tests/ -v
```

The suite covers albedo-anomaly math (incl. DOY wrap-around at year end), the
dual-criterion and visibility-only labels, the next-day label shift,
train-fold-only imputation, temporal-leakage-free CV splits, F₂-optimal
threshold tuning, per-fold PR-AUC/ROC-AUC, the paired bootstrap CI for ΔPR-AUC,
per-station PR-AUC/F₂, the **feature-group assignment and driver ablation**
(verifying the ablation recovers a known driver above noise), the Wilcoxon
statistics, a check that the meteorological baseline has real 24-hour skill, the
keyless real-source parsers (Open-Meteo aggregation incl.
wind-direction decomposition, NOAA ISD visibility parsing, the Liang albedo /
NDVI computation — all network-mocked so CI stays offline), and an end-to-end
smoke test on the bundled synthetic data. CI (GitHub Actions) runs the tests on
Python 3.10–3.12 plus a full synthetic pipeline run, and uploads the report and
figures as artifacts.

## Model & Drivers

### Features
ERA5 daily aggregates — wind speed/gust, **wind-direction components (the NW shamal: `wind_n_mean`, `wind_e_mean`, `northerly_frac`)**, RH/VPD, boundary-layer height, soil moisture/temperature, precipitation (incl. 7-day antecedent) — plus lag/diff features, MODIS **NDVI** and shortwave **reflectivity** indices, static SoilGrids properties (clay, sand, silt, OCS, bulk density), and cyclical day-of-year / month encodings.

### Driver groups (for the ablation)
Features are bucketed into physical driver groups via `build_feature_groups`, and the ablation drops each in turn to measure its incremental skill:

`wind_speed`, `wind_direction`, `humidity_dryness`, `antecedent_moisture`, `thermal_blh`, `pressure`, `vegetation`, `soil_texture`, `seasonality`, `albedo`.

### Label definition
A **dust event on day D+1** is predicted from features on day D.
- **Synthetic / GEE path** (`label_mode="dual"`): confirmed event requires **both** MOD09A1 NDDI > 0 (8-day composite, forward-filled) **and** at least one hourly visibility ≤ 1 000 m.
- **Keyless real path** (`label_mode="visibility"`): the **WMO dust-storm criterion** — at least one hourly visibility ≤ 1 000 m (NOAA ISD). The 8-day MODIS NDDI is too coarse and numerically unstable over bright desert to gate individual dust days, so it is dropped from labeling (NDVI is retained as a feature).

### Evaluation
- **Primary metric:** PR-AUC (average precision) — threshold-independent, the standard for rare-event detection. **Secondary:** ROC-AUC. **Operational:** F₂-score (β=2) at a tuned threshold.
- **CV:** `TimeSeriesSplit` (8 folds) on data sorted by **date** across stations, so every training fold strictly precedes its test fold in time (no temporal leakage)
- **Decision threshold** (F₂ only): tuned to maximise F₂ on a held-out validation slice (last 15%) of each training fold, then applied to the test fold rather than using a fixed 0.5 cut-off. PR-AUC/ROC-AUC are threshold-free and so unaffected by this.
- **Optional:** `--station-cv` for leave-one-station-out `GroupKFold`
- **Naive baselines:** no-skill (base rate), persistence, and a meteorology-only model contextualise the skill (EMBRACE-style)
- **Driver ablation:** for each driver group, retrain on all-features-minus-group and measure the incremental PR-AUC with a paired **bootstrap 95% CI** and a two-sided bootstrap p-value; p-values are **Benjamini–Hochberg FDR-corrected** across the 10 groups
- **Robustness:** model PR-AUC/ROC-AUC and the top driver's Δ are re-estimated over 5 seeds (mean ± sd); a **reliability/calibration** diagram and **per-station** breakdown are produced

## Real Data Mode (keyless — no accounts required)

The default real-data path uses **only public APIs that need no account, key, or
OAuth**, so it runs anywhere with outbound network access:

```bash
python run_pipeline.py --mode real
```

No extra dependencies beyond `requirements.txt` — everything is fetched with
`requests`. Downloads are cached under `data/raw/real/`, so re-runs are
incremental.

| Variable | Source | Endpoint | Key? |
|----------|--------|----------|------|
| ERA5 meteorology | **Open-Meteo** Historical Weather API | `archive-api.open-meteo.com` | No |
| MODIS reflectance → NDVI, NDDI, **albedo** | **ORNL DAAC** MODIS subsets (MOD09A1) | `modis.ornl.gov/rst` | No |
| Station visibility | **NOAA ISD** global-hourly CSV | `ncei.noaa.gov/data/global-hourly` | No |
| Soil properties | **ISRIC SoilGrids** v2 | `rest.isric.org/soilgrids` | No |

### How the MODIS features are derived (keyless)
ORNL serves MODIS **surface reflectance** (MOD09A1, 8-day, 500 m) globally
without login. The spatial mean over a ±`albedo_km` box yields **NDVI** and a
shortwave **reflectivity / albedo** index (via the **Liang (2001)**
narrow-to-broadband coefficients, bands 1–5, 7), each turned into a
DOY-climatology anomaly. (The optional GEE path uses native MODIS/061 products;
the keyless ORNL path is what lets the project run with no credentials.)

### Scope and cost
MODIS is **one ORNL request per band per 8-day composite**, so the live run is
bounded by `config.yaml → real:` (default 3 stations, 2018–2020 study, 2017–2020
MODIS, ±20 km box). Scale up with:

```bash
python run_pipeline.py --mode real \
  --stations riyadh hafar sharurah dammam tabuk qassim arar najran \
  --study-years 2018 2019 2020 2021 2022 \
  --modis-years 2016 2017 2018 2019 2020 2021 2022 \
  --albedo-km 40
```

Real-mode results are written to `results/report_real.md` and
`data/final/master_dataset_real.csv`.

### Optional: Google Earth Engine + Copernicus CDS path
For native MCD43A3 albedo and the ERA5 archive, install the extras and
authenticate; these functions live in `src/acquisition.py`:

```bash
pip install -r requirements-real.txt   # earthengine-api, cdsapi, isd, netCDF4
earthengine authenticate               # GEE
# ~/.cdsapirc with your Copernicus CDS UID:key
```

## CLI Options

```bash
python run_pipeline.py                  # synthetic mode (default)
python run_pipeline.py --mode real      # keyless real data (Open-Meteo/ORNL/NOAA/SoilGrids)
python run_pipeline.py --mode real --stations riyadh hafar --study-years 2019 2020
python run_pipeline.py --mode real --albedo-km 40   # wider MODIS footprint
python run_pipeline.py --tune           # Optuna hyperparameter search (slower)
python run_pipeline.py --station-cv     # add leave-one-station-out evaluation
python run_pipeline.py --config my.yaml # custom config path
```

## Key Implementation Details

### Anomaly features
Satellite indices (NDVI, reflectivity/albedo) are converted to anomalies relative
to a per-day-of-year climatology:
```
anomaly(t) = value(t) - mean_baseline(DOY ± 15 days)
```
Baseline years: 2013–2017 (5-year climatology). Study period: 2018–2022.

### Class imbalance
`scale_pos_weight = n_negative / n_positive` computed **per training fold** only, combined with F₂-optimal threshold selection on a within-fold validation slice.

### Temporal leakage prevention
For `TimeSeriesSplit`, data is sorted by `[date, station]` before splitting so every training row precedes every test row in time. Labels are shifted with `shift(-1)` so features on day D predict day D+1, and fold-wise median imputation and `scale_pos_weight` are computed on the training fold only.

## Outputs

After a successful run:

| File | Description |
|------|-------------|
| `results/report.md` / `report_real.md` | Model performance, driver ablation, per-station, conclusion |
| `outputs/driver_ablation.png` | **Incremental PR-AUC by driver group (the headline chart)** |
| `outputs/driver_ablation.csv` | Ablation table (incremental PR-AUC + CIs) |
| `outputs/pr_curve.png` | Precision–recall curve for the forecast model |
| `outputs/shap_importance.png` | Top-20 SHAP features |
| `data/final/master_dataset[_real].csv` | Merged feature matrix |
| `outputs/real/…` | Same figures for the real-data run |

## Requirements

- Python 3.9+
- See `requirements.txt` for packages (`xgboost`, `pandas`, `xarray`, `scikit-learn`, `scipy`, `shap`, `optuna`, etc.)

## Interactive demo & paper

- **Live demo:** `streamlit run app.py` — explore the driver ablation, a what-if
  dust-risk predictor (move wind/humidity/NDVI/albedo sliders for a live
  probability), and per-station forecasts. One-click deploy to
  [Streamlit Community Cloud](https://streamlit.io/cloud) for a public URL.
  Install with `pip install -r requirements-demo.txt`.
- **Project page (GitHub Pages):** [`docs/index.html`](docs/index.html) — a
  static, self-contained dashboard with the findings, figures, and an in-browser
  dust-risk calculator (deployed by `.github/workflows/pages.yml`).
- **Short paper:** [`PAPER.md`](PAPER.md) — abstract, methods, results,
  discussion, limitations.

## Citation & Background

Methodology and data sources:
- **Albedo (keyless path):** shortwave broadband albedo from MODIS MOD09A1 surface
  reflectance via the **Liang (2001)** narrow-to-broadband conversion. *(The
  optional GEE path uses native MODIS/061 MCD43A3 white-sky albedo.)*
- **Vegetation / dust index:** NDVI (and NDDI, Qu et al. 2006) from MOD09A1.
- **Meteorology:** ERA5 reanalysis via the keyless Open-Meteo archive *(or
  Copernicus CDS in the GEE path)*.
- **Visibility:** NOAA Integrated Surface Database (WMO dust-storm criterion).
- **Soil:** ISRIC SoilGrids v2 static properties.

References: Liang, S. (2001), *Narrowband to broadband conversions of land surface
albedo*, Remote Sens. Environ. 76(2). Qu et al. (2006), NDDI. Hersbach et al.
(2020), ERA5.

## License

MIT License — see repository for details.
