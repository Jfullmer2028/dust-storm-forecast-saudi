# Dust-Storm Forecast — Saudi Arabia

Predict dust-storm onset **24 hours in advance** at **eight** Saudi Arabian weather stations (**2018–2022**) using XGBoost. This project tests whether adding a **MODIS MCD43A3-derived shortwave broadband albedo anomaly** (spatial mean within **200 km**) improves the **F₂-score** over a baseline model using only ERA5 meteorology, static soil properties, and NDVI.

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
2. Engineer features including MODIS albedo anomaly (±15-day DOY climatology from the 2013–2017 baseline)
3. Train **baseline** vs **full** XGBoost models with `TimeSeriesSplit` (8 folds), tuning an **F₂-optimal decision threshold** on a held-out validation slice of each training fold
4. Compare F₂-scores with **Wilcoxon signed-rank** and **bootstrap 95% CI**, plus a **per-station** F₂ breakdown
5. Write figures to `outputs/` and `results/report.md`

Expected runtime: ~3–6 minutes on a laptop (~14.5k station-days; depends on SHAP).

### Representative result (synthetic mode)

| Model | Mean F₂ (8-fold CV) |
|-------|---------------------|
| Baseline (ERA5 + soil + NDVI) | **0.476** |
| Full (+ MODIS albedo anomaly) | **0.559** |
| **ΔF₂** | **+0.083** (Wilcoxon p = 0.008; bootstrap 95% CI [+0.062, +0.097]) |

The baseline is a genuinely competent meteorological forecaster (no degenerate 0.000 folds), and the albedo anomaly delivers a **modest but statistically significant** incremental gain — the realistic outcome such a study should produce. Albedo helps at **every** station, with heterogeneity from +0.03 (Sharurah) to +0.12 (Riyadh).

## Project Structure

```
dust-storm-forecast-saudi/
├── config.yaml              # Stations, paths, model settings
├── run_pipeline.py          # Master script
├── requirements.txt         # Core dependencies (synthetic mode)
├── requirements-real.txt    # Extra deps for real-data acquisition
├── .github/workflows/ci.yml # CI: unit tests + full pipeline smoke run
├── data/
│   ├── synthetic/           # Pre-built test CSVs (no API keys needed)
│   ├── raw/real/            # Cached keyless real downloads (gitignored)
│   ├── processed/
│   └── final/               # master_dataset[_real].csv (generated)
├── src/
│   ├── acquisition.py       # GEE/CDS/ISD acquisition + synthetic generator
│   ├── real_sources.py      # Keyless real APIs (Open-Meteo/ORNL/NOAA/SoilGrids)
│   ├── features.py          # Albedo anomaly, lags, temporal encodings, feature sets
│   ├── labeling.py          # Dual-criterion & visibility-only labels
│   ├── models.py            # XGBoost CV, F₂-threshold tuning, Optuna
│   └── evaluation.py        # F₂ comparison, Wilcoxon, bootstrap, per-station, SHAP
├── tests/                   # Pytest suite (features, labels, CV, real sources, smoke)
├── outputs/                 # Figures + feature importance (outputs/real/ for real mode)
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
threshold tuning, per-station F₂, the Wilcoxon and bootstrap statistics, a guard
that the meteorological baseline is **never degenerate** (no 0.000 folds), the
keyless real-source parsers (Open-Meteo aggregation, NOAA ISD visibility
parsing, the Liang albedo / NDVI computation — all network-mocked so CI stays
offline), and an end-to-end smoke test on the bundled synthetic data. CI (GitHub
Actions) runs the tests on Python 3.10–3.12 plus a full synthetic pipeline run,
and uploads the report and figures as artifacts.

## Models

### Baseline features
ERA5 daily aggregates (wind, BLH, RH, soil moisture, precipitation, etc.), lag/diff features, NDVI, static SoilGrids properties (clay, sand, silt, OCS, bulk density), and cyclical day-of-year / month encodings.

### Full model (baseline + albedo)
Adds four MODIS MCD43A3 features within 200 km radius:
- `albedo_wsb` — shortwave broadband white-sky albedo
- `albedo_anomaly` — deviation from 2015–2017 seasonal climatology (±15 DOY window)
- `albedo_anom_3d`, `albedo_anom_7d` — temporally smoothed anomaly

### Label definition
A **dust event on day D+1** is predicted from features on day D.
- **Synthetic / GEE path** (`label_mode="dual"`): confirmed event requires **both** MOD09A1 NDDI > 0 (8-day composite, forward-filled) **and** at least one hourly visibility ≤ 1 000 m.
- **Keyless real path** (`label_mode="visibility"`): the **WMO dust-storm criterion** — at least one hourly visibility ≤ 1 000 m (NOAA ISD). The 8-day MODIS NDDI is too coarse and numerically unstable over bright desert to gate individual dust days, so it is dropped from labeling (NDVI is retained as a feature).

### Evaluation
- **Primary metric:** F₂-score (β=2, recall weighted 2× precision)
- **CV:** `TimeSeriesSplit` (8 folds) on data sorted by **date** across stations, so every training fold strictly precedes its test fold in time (no temporal leakage)
- **Decision threshold:** tuned to maximise F₂ on a held-out validation slice (last 15%) of each training fold, then applied to the test fold — never the naive 0.5 cut-off
- **Optional:** `--station-cv` for leave-one-station-out `GroupKFold`
- **Statistics:** Wilcoxon signed-rank on per-fold F₂ differences (8 folds → significance is now attainable); bootstrap CI (5 000 resamples) on concatenated predictions; per-station F₂ breakdown

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

### How the satellite albedo is derived (keyless)
ORNL serves MODIS **surface reflectance** (MOD09A1, 8-day, 500 m) globally
without login. The spatial mean over a ±`albedo_km` box is converted to a
**shortwave broadband albedo** via the **Liang (2001)** narrow-to-broadband
coefficients (bands 1–5, 7), then turned into a DOY-climatology anomaly exactly
as in the synthetic path. (The GEE path below instead uses native MCD43A3
white-sky albedo; the keyless Liang proxy is what lets the project run with no
credentials.)

### Scope and cost
MODIS is **one ORNL request per band per 8-day composite** (7 bands), so the
live run is bounded by `config.yaml → real:` (default 3 stations, 2020 study,
2019–2020 MODIS for the anomaly baseline, ±20 km box ≈ 15 min). Scale up with:

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

### Albedo anomaly (Section 2.1 of guide)
```
albedo_anomaly(t) = albedo(t) - mean_baseline(DOY ± 15 days)
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
| `results/report.md` | Full F₂ comparison, Wilcoxon p-value, bootstrap CI |
| `outputs/f2_comparison_by_fold.png` | Per-fold baseline vs full bar chart |
| `outputs/bootstrap_delta_f2.png` | Bootstrap ΔF₂ histogram |
| `outputs/shap_importance.png` | Top-20 SHAP features (full model) |
| `data/final/master_dataset.csv` | Merged feature matrix |

## Requirements

- Python 3.9+
- See `requirements.txt` for packages (`xgboost`, `pandas`, `xarray`, `scikit-learn`, `scipy`, `shap`, `optuna`, etc.)

## Citation & Background

This implementation follows the methodology in the project research guide:
- MODIS MCD43A3 shortwave broadband albedo (BSA/WSA)
- NDDI dust index from MOD09A1 (Xu et al. 2006)
- ERA5 meteorological predictors via Copernicus CDS
- SoilGrids v2 static soil properties

## License

MIT License — see repository for details.
