#!/usr/bin/env python3
"""
Master pipeline: data acquisition -> features -> labels -> XGBoost CV -> statistics.

Usage:
  python run_pipeline.py                    # synthetic data (default)
  python run_pipeline.py --mode real        # requires GEE, CDS, ISD credentials
  python run_pipeline.py --tune             # enable Optuna hyperparameter search
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.acquisition import (  # noqa: E402
    generate_all_synthetic_data,
    load_config,
    load_synthetic_station_bundle,
)
from src.evaluation import (  # noqa: E402
    evaluate_both_models,
    full_statistical_analysis,
    plot_feature_importance,
    run_group_ablation,
    write_report,
)
from src.features import (  # noqa: E402
    BASELINE_FEATURES,
    FULL_FEATURES,
    REAL_BASELINE_FEATURES,
    REAL_FULL_FEATURES,
    compute_albedo_anomaly,
)
from src.labeling import build_full_dataset, build_master_dataframe  # noqa: E402
from src.models import tune_xgboost  # noqa: E402


def build_dataset_from_synthetic(config: dict) -> "pd.DataFrame":
    """Generate/load synthetic CSVs and assemble master dataset."""
    import pandas as pd

    syn_dir = Path(config["paths"]["data_synthetic"])
    final_path = Path(config["paths"]["data_final"]) / "master_dataset.csv"

    if not (syn_dir / "soil_properties.csv").exists():
        print("Generating synthetic test dataset...")
        generate_all_synthetic_data(config, syn_dir)
    else:
        print(f"Using existing synthetic data in {syn_dir}")

    station_dfs = []
    for station_name in config["stations"]:
        bundle = load_synthetic_station_bundle(station_name, syn_dir)

        albedo_anom = compute_albedo_anomaly(
            bundle["albedo"],
            baseline_years=config["baseline_years"],
            study_years=config["study_years"],
            doy_window=config["project"]["albedo_doy_window"],
        )

        station_df = build_master_dataframe(
            station_name,
            bundle["era5"],
            albedo_anom,
            bundle["mod09"],
            bundle["vis_flag"],
            bundle["soil_feats"],
            study_start=config["project"]["study_start"][:4],
            study_end=config["project"]["study_end"][:4],
        )
        station_dfs.append(station_df)
        print(
            f"  {station_name}: {len(station_df)} rows, "
            f"{station_df['dust_event_next_day'].sum()} dust events"
        )

    master = build_full_dataset(station_dfs)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(final_path, index=False)
    print(f"Master dataset saved: {final_path} ({len(master)} rows)")
    return master


def build_dataset_from_real(config: dict, args: argparse.Namespace) -> "pd.DataFrame":
    """
    Acquire real data from keyless public APIs and assemble the master dataset.

    Sources (no account / key / OAuth required):
      - Meteorology : Open-Meteo Historical Weather API (ERA5 reanalysis)
      - Satellite   : ORNL DAAC MODIS MOD09A1 -> NDVI, NDDI, Liang albedo
      - Visibility  : NOAA ISD global-hourly CSV archive
      - Soil        : ISRIC SoilGrids v2 REST

    All downloads are cached under data/raw/real so re-runs are incremental.
    """
    from src.features import REAL_FULL_FEATURES
    from src.real_sources import fetch_station_real_data

    real_cfg = config.get("real", {})
    stations_all = config["stations"]

    station_names = args.stations or real_cfg.get("stations", list(stations_all))
    study_years = args.study_years or real_cfg.get("study_years", [2020])
    modis_years = args.modis_years or real_cfg.get("modis_years", study_years)
    modis_years = sorted(set(modis_years) | set(study_years))
    albedo_km = args.albedo_km or real_cfg.get("albedo_km", 20)
    baseline_years = [y for y in modis_years if y not in study_years] or modis_years

    cache_dir = Path(config["paths"]["data_raw"]) / "real"

    print(
        f"Real-data scope: stations={station_names}  study={study_years}  "
        f"modis={modis_years}  albedo_km=+/-{albedo_km}"
    )

    station_dfs = []
    for name in station_names:
        coords = stations_all[name]
        isd_id = {"usaf": coords["isd_usaf"], "wban": coords["isd_wban"]}
        bundle = fetch_station_real_data(
            name,
            coords,
            isd_id,
            study_years=study_years,
            modis_years=modis_years,
            soil_properties=config["soil_properties"],
            albedo_km=albedo_km,
            cache_dir=cache_dir,
        )

        albedo_anom = compute_albedo_anomaly(
            bundle["albedo"],
            baseline_years=baseline_years,
            study_years=study_years,
            doy_window=config["project"]["albedo_doy_window"],
        )

        station_df = build_master_dataframe(
            name,
            bundle["era5"],
            albedo_anom,
            bundle["mod09"],
            bundle["vis_flag"],
            bundle["soil_feats"],
            study_start=str(min(study_years)),
            study_end=str(max(study_years)),
            label_mode="visibility",
        )
        station_dfs.append(station_df)
        print(
            f"  {name}: {len(station_df)} rows, "
            f"{int(station_df['dust_event_next_day'].sum())} dust events"
        )

    master = build_full_dataset(station_dfs)
    # Keep only real features that were actually produced (guards optional cols).
    present = [c for c in REAL_FULL_FEATURES if c in master.columns]
    missing = [c for c in REAL_FULL_FEATURES if c not in master.columns]
    if missing:
        print(f"  Note: {len(missing)} real features unavailable: {missing}")

    final_path = Path(config["paths"]["data_final"]) / "master_dataset_real.csv"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(final_path, index=False)
    print(f"Real master dataset saved: {final_path} ({len(master)} rows)")
    return master


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dust-storm onset prediction pipeline (Saudi Arabia)"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=["synthetic", "real"],
        default=None,
        help="Data source (default: synthetic if config synthetic.enabled)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter tuning on full model",
    )
    parser.add_argument(
        "--station-cv",
        action="store_true",
        help="Also run leave-one-station-out GroupKFold CV",
    )
    parser.add_argument(
        "--stations",
        nargs="+",
        default=None,
        help="Subset of station names (real mode); overrides config real.stations",
    )
    parser.add_argument(
        "--study-years",
        nargs="+",
        type=int,
        default=None,
        help="Study years for real mode (e.g. --study-years 2019 2020)",
    )
    parser.add_argument(
        "--modis-years",
        nargs="+",
        type=int,
        default=None,
        help="MODIS years for real mode (extra years form the albedo baseline)",
    )
    parser.add_argument(
        "--albedo-km",
        type=int,
        default=None,
        help="+/- box half-width (km) for the MODIS spatial mean (real mode)",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)

    mode = args.mode
    if mode is None:
        mode = "synthetic" if config.get("synthetic", {}).get("enabled", True) else "real"

    print("=" * 60)
    print("DUST-STORM ONSET PREDICTION PIPELINE")
    print(f"Mode: {mode}")
    print("=" * 60)

    # --- Step 1: Build dataset ---
    print("\n[1/4] Building master dataset...")
    if mode == "synthetic":
        df = build_dataset_from_synthetic(config)
        baseline_features = BASELINE_FEATURES
        full_features = FULL_FEATURES
        n_splits = config["model"]["n_cv_splits"]
        output_dir = PROJECT_ROOT / config["paths"]["outputs"]
        report_path = PROJECT_ROOT / config["paths"]["results"] / "report.md"
        dataset_name = "master_dataset.csv"
    else:
        df = build_dataset_from_real(config, args)
        # Use only the real features that were actually produced.
        baseline_features = [c for c in REAL_BASELINE_FEATURES if c in df.columns]
        full_features = [c for c in REAL_FULL_FEATURES if c in df.columns]
        n_splits = config.get("real", {}).get(
            "n_cv_splits", config["model"]["n_cv_splits"]
        )
        output_dir = PROJECT_ROOT / config["paths"]["outputs"] / "real"
        report_path = PROJECT_ROOT / config["paths"]["results"] / "report_real.md"
        dataset_name = "master_dataset_real.csv"

    random_state = config["model"]["random_state"]
    results_dir = PROJECT_ROOT / config["paths"]["results"]

    xgb_params = None
    if args.tune or config["model"].get("use_optuna", False):
        print("\n[Optuna] Tuning hyperparameters on full feature set...")
        xgb_params = tune_xgboost(
            df,
            full_features,
            n_trials=config["model"]["optuna_trials"],
            random_state=random_state,
        )
        print(f"  Best params: {xgb_params}")

    # --- Step 2: Cross-validation ---
    print("\n[2/4] Running time-series cross-validation...")
    baseline_results, full_results = evaluate_both_models(
        df,
        n_splits=n_splits,
        output_dir=output_dir,
        xgb_params=xgb_params,
        random_state=random_state,
        baseline_features=baseline_features,
        full_features=full_features,
    )

    if args.station_cv:
        from src.models import run_cross_validation

        print("\n[Optional] Leave-one-station-out CV (baseline)...")
        run_cross_validation(
            df, baseline_features, cv_strategy="station", random_state=random_state
        )
        print("\n[Optional] Leave-one-station-out CV (full)...")
        run_cross_validation(
            df, full_features, cv_strategy="station", random_state=random_state
        )

    # --- Step 3: Statistical comparison ---
    print("\n[3/4] Statistical comparison (Wilcoxon + bootstrap)...")
    stats = full_statistical_analysis(
        baseline_results,
        full_results,
        n_bootstrap=config["model"]["n_bootstrap"],
        confidence_level=config["model"]["confidence_level"],
        random_state=random_state,
        output_dir=output_dir,
    )

    # --- Step 4: Driver ablation (what actually matters) ---
    print("\n[4/5] Driver ablation (incremental PR-AUC by group)...")
    ablation_df = run_group_ablation(
        df,
        full_features,
        n_splits=n_splits,
        xgb_params=xgb_params,
        random_state=random_state,
        n_bootstrap=2000,
        output_dir=output_dir,
    )

    # --- Step 5: Feature importance ---
    print("\n[5/5] SHAP feature importance (full model)...")
    plot_feature_importance(
        df,
        full_features,
        output_dir=output_dir,
        random_state=random_state,
    )

    write_report(
        stats,
        baseline_results,
        full_results,
        df,
        output_path=report_path,
        data_mode=mode,
        ablation_df=ablation_df,
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  Dataset:  {config['paths']['data_final']}/{dataset_name}")
    print(f"  Figures:  {output_dir}/")
    print(f"  Report:   {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
