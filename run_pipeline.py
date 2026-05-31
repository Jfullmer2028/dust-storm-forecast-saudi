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
    write_report,
)
from src.features import FULL_FEATURES, compute_albedo_anomaly  # noqa: E402
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


def build_dataset_from_real(config: dict) -> "pd.DataFrame":
    """
    Acquire real data from NOAA ISD, GEE, CDS, SoilGrids.

    Requires:
      - earthengine authenticate (ee.Authenticate())
      - ~/.cdsapirc for ERA5
      - pip install isd earthengine-api cdsapi
    """
    from src.acquisition import (
        build_soil_dataframe,
        daily_visibility_flag,
        download_era5,
        download_isd_visibility,
        era5_to_daily_features,
        get_albedo_timeseries,
        get_mod09a1_timeseries,
    )

    stations = config["stations"]
    raw_dir = Path(config["paths"]["data_raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Soil (cached)
    soil_path = raw_dir / "soil_properties.csv"
    if soil_path.exists():
        import pandas as pd

        soil_df = pd.read_csv(soil_path, index_col="station")
    else:
        print("Fetching SoilGrids properties...")
        soil_df = build_soil_dataframe(stations, config["soil_properties"])
        soil_df.to_csv(soil_path)

    station_dfs = []
    study_years = config["study_years"]
    baseline_start = f"{min(config['baseline_years'])}-01-01"
    study_end = config["project"]["study_end"]

    for name, coords in stations.items():
        print(f"\n--- {name} ---")

        # Visibility
        vis_parts = []
        for year in study_years:
            vis_path = raw_dir / f"vis_{name}_{year}.csv"
            if vis_path.exists():
                import pandas as pd

                vis_parts.append(
                    pd.read_csv(vis_path, index_col="datetime", parse_dates=True)
                )
            else:
                print(f"  Downloading ISD visibility {year}...")
                vis = download_isd_visibility(
                    coords["wmo_id"], coords["wban"], year
                )
                vis.to_csv(vis_path)
                vis_parts.append(vis)
        import pandas as pd

        vis_all = pd.concat(vis_parts).sort_index()
        vis_flag = daily_visibility_flag(
            vis_all, config["project"]["visibility_threshold_m"]
        )

        # ERA5
        era5_nc = raw_dir / f"era5_{name}.nc"
        era5_csv = raw_dir / f"era5_{name}_daily.csv"
        if era5_csv.exists():
            era5_df = pd.read_csv(era5_csv, index_col="date", parse_dates=True)
        else:
            if not era5_nc.exists():
                print("  Downloading ERA5 (this may take a while)...")
                download_era5(
                    coords["lat"],
                    coords["lon"],
                    study_years,
                    era5_nc,
                    config["era5_variables"],
                )
            era5_df = era5_to_daily_features(era5_nc, coords["lat"], coords["lon"])
            era5_df.to_csv(era5_csv)

        # Albedo
        alb_path = raw_dir / f"albedo_{name}.csv"
        if alb_path.exists():
            albedo_raw = pd.read_csv(alb_path, index_col="date", parse_dates=True)
        else:
            print("  Extracting MCD43A3 albedo from GEE...")
            albedo_raw = get_albedo_timeseries(
                name,
                stations,
                radius_m=config["project"]["albedo_radius_m"],
                start=baseline_start,
                end=study_end,
            )
            albedo_raw.to_csv(alb_path)

        albedo_anom = compute_albedo_anomaly(
            albedo_raw,
            baseline_years=config["baseline_years"],
            study_years=config["study_years"],
            doy_window=config["project"]["albedo_doy_window"],
        )

        # MOD09A1
        mod_path = raw_dir / f"mod09a1_{name}.csv"
        if mod_path.exists():
            mod09 = pd.read_csv(mod_path, index_col="date", parse_dates=True)
        else:
            print("  Extracting MOD09A1 NDVI/NDDI from GEE...")
            mod09 = get_mod09a1_timeseries(
                name,
                stations,
                radius_m=config["project"]["albedo_radius_m"],
                start=config["project"]["study_start"],
                end=study_end,
            )
            mod09.to_csv(mod_path)

        soil_feats = soil_df.loc[name].to_dict()
        station_df = build_master_dataframe(
            name,
            era5_df,
            albedo_anom,
            mod09,
            vis_flag,
            soil_feats,
            study_start=str(min(study_years)),
            study_end=str(max(study_years)),
        )
        station_dfs.append(station_df)

    master = build_full_dataset(station_dfs)
    final_path = Path(config["paths"]["data_final"]) / "master_dataset.csv"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(final_path, index=False)
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
    else:
        df = build_dataset_from_real(config)

    n_splits = config["model"]["n_cv_splits"]
    random_state = config["model"]["random_state"]
    output_dir = PROJECT_ROOT / config["paths"]["outputs"]
    results_dir = PROJECT_ROOT / config["paths"]["results"]

    xgb_params = None
    if args.tune or config["model"].get("use_optuna", False):
        print("\n[Optuna] Tuning hyperparameters on full feature set...")
        xgb_params = tune_xgboost(
            df,
            FULL_FEATURES,
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
    )

    if args.station_cv:
        from src.features import BASELINE_FEATURES
        from src.models import run_cross_validation

        print("\n[Optional] Leave-one-station-out CV (baseline)...")
        run_cross_validation(
            df, BASELINE_FEATURES, cv_strategy="station", random_state=random_state
        )
        print("\n[Optional] Leave-one-station-out CV (full)...")
        run_cross_validation(
            df, FULL_FEATURES, cv_strategy="station", random_state=random_state
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

    # --- Step 4: Feature importance ---
    print("\n[4/4] SHAP feature importance (full model)...")
    plot_feature_importance(
        df,
        FULL_FEATURES,
        output_dir=output_dir,
        random_state=random_state,
    )

    write_report(
        stats,
        baseline_results,
        full_results,
        df,
        output_path=results_dir / "report.md",
        data_mode=mode,
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  Dataset:  {config['paths']['data_final']}/master_dataset.csv")
    print(f"  Figures:  {config['paths']['outputs']}/")
    print(f"  Report:   {config['paths']['results']}/report.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
