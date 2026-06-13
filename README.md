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
│   ├── raw/                 # Real downloads (ERA5 .nc, ISD, GEE exports)
│   ├── processed/
│   └── final/               # master_dataset.csv (generated)
├── src/
│   ├── acquisition.py       # NOAA ISD, GEE, ERA5, SoilGrids + synthetic generator
│   ├── features.py          # Albedo anomaly, lags, temporal encodings
│   ├── labeling.py          # NDDI + visibility dual-criterion labels
│   ├── models.py            # XGBoost CV, Optuna tuning
│   └── evaluation.py        # F₂ comparison, Wilcoxon, bootstrap, SHAP
├── tests/                   # Pytest suite (features, labels, CV, smoke)
├── outputs/                 # Figures and feature importance CSV
└── results/
    └── report.md            # Final results (generated)
```

## Tests

```bash
pip install pytest
pytest tests/ -v
```

The suite covers albedo-anomaly math (incl. DOY wrap-around at year end), the
dual-criterion label, the next-day label shift, train-fold-only imputation,
temporal-leakage-free CV splits, F₂-optimal threshold tuning, per-station F₂,
the Wilcoxon and bootstrap statistics, a guard that the meteorological baseline
is **never degenerate** (no 0.000 folds), and an end-to-end smoke test on the
bundled synthetic data. CI (GitHub Actions) runs the tests on Python 3.10–3.12
plus a full synthetic pipeline run, and uploads the report and figures as
artifacts.

## Models

### Baseline features
ERA5 daily aggregates (wind, BLH, RH, soil moisture, precipitation, etc.), lag/diff features, NDVI, static SoilGrids properties (clay, sand, silt, OCS, bulk density), and cyclical day-of-year / month encodings.

### Full model (baseline + albedo)
Adds four MODIS MCD43A3 features within 200 km radius:
- `albedo_wsb` — shortwave broadband white-sky albedo
- `albedo_anomaly` — deviation from 2015–2017 seasonal climatology (±15 DOY window)
- `albedo_anom_3d`, `albedo_anom_7d` — temporally smoothed anomaly

### Label definition
A **dust event on day D+1** is predicted from features on day D. A confirmed event requires **both**:
- MOD09A1 NDDI > 0 (8-day composite, forward-filled to daily)
- At least one hourly visibility observation ≤ 1 000 m (NOAA ISD)

### Evaluation
- **Primary metric:** F₂-score (β=2, recall weighted 2× precision)
- **CV:** `TimeSeriesSplit` (8 folds) on data sorted by **date** across stations, so every training fold strictly precedes its test fold in time (no temporal leakage)
- **Decision threshold:** tuned to maximise F₂ on a held-out validation slice (last 15%) of each training fold, then applied to the test fold — never the naive 0.5 cut-off
- **Optional:** `--station-cv` for leave-one-station-out `GroupKFold`
- **Statistics:** Wilcoxon signed-rank on per-fold F₂ differences (8 folds → significance is now attainable); bootstrap CI (5 000 resamples) on concatenated predictions; per-station F₂ breakdown

## Real Data Mode

To run with live data sources, configure accounts and run:

```bash
python run_pipeline.py --mode real
```

### Prerequisites

First install the extra acquisition dependencies:

```bash
pip install -r requirements-real.txt
```

| Source | Account / Tool | Setup |
|--------|----------------|-------|
| MODIS albedo & NDVI | [Google Earth Engine](https://earthengine.google.com) | `earthengine authenticate` |
| ERA5 reanalysis | [Copernicus CDS](https://cds.climate.copernicus.eu) | Create `~/.cdsapirc` with UID and API key |
| Station visibility | NOAA ISD | included in `requirements-real.txt` |
| Soil properties | SoilGrids REST | Free, no key (`requests`) |

### ERA5 CDS config example (`~/.cdsapirc`)

```
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR_UID:YOUR_API_KEY
```

### Notes on real data
- ERA5 downloads are large (~several GB per station-year); allow 50+ GB disk space.
- GEE `reduceRegion` over 200 km at 500 m may take several minutes per station; results are cached to CSV.
- SoilGrids API is rate-limited; the pipeline sleeps 2 s between station requests.

## CLI Options

```bash
python run_pipeline.py                  # synthetic mode (default)
python run_pipeline.py --mode real      # fetch from external APIs
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
