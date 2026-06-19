"""End-to-end smoke test on the bundled synthetic data (no network)."""

from pathlib import Path

import pytest

from src.acquisition import load_config, load_synthetic_station_bundle
from src.evaluation import run_group_ablation
from src.features import BASELINE_FEATURES, FULL_FEATURES, compute_albedo_anomaly
from src.labeling import build_full_dataset, build_master_dataframe
from src.models import run_cross_validation

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def master_df():
    config = load_config(PROJECT_ROOT / "config.yaml")
    syn_dir = PROJECT_ROOT / config["paths"]["data_synthetic"]
    if not (syn_dir / "soil_properties.csv").exists():
        pytest.skip("synthetic data not generated")

    station_dfs = []
    for station_name in config["stations"]:
        bundle = load_synthetic_station_bundle(station_name, syn_dir)
        albedo_anom = compute_albedo_anomaly(
            bundle["albedo"],
            baseline_years=config["baseline_years"],
            study_years=config["study_years"],
            doy_window=config["project"]["albedo_doy_window"],
        )
        station_dfs.append(
            build_master_dataframe(
                station_name,
                bundle["era5"],
                albedo_anom,
                bundle["mod09"],
                bundle["vis_flag"],
                bundle["soil_feats"],
                study_start="2018",
                study_end="2022",
            )
        )
    return build_full_dataset(station_dfs)


def test_master_dataset_shape_and_labels(master_df):
    assert len(master_df) > 10000  # 8 stations x 5 study years
    assert master_df["station"].nunique() == 8
    positives = master_df["dust_event_next_day"].sum()
    assert 0 < positives < len(master_df) * 0.5  # imbalanced but non-empty
    assert set(FULL_FEATURES) <= set(master_df.columns)


def test_baseline_has_real_skill(master_df):
    """The meteorological baseline has genuine 24h skill on every fold."""
    baseline = run_cross_validation(
        master_df,
        BASELINE_FEATURES,
        n_splits=5,
        xgb_params={"n_estimators": 120},
        verbose=False,
    )
    # Synoptic persistence gives the baseline a real next-day signal.
    assert baseline["mean_f2"] > 0.20
    assert (baseline["fold_f2"] > 0.0).all()


def test_forecast_model_has_skill(master_df):
    """The forecast model ranks dust risk well above the base rate."""
    res = run_cross_validation(
        master_df, FULL_FEATURES, n_splits=6,
        xgb_params={"n_estimators": 120}, verbose=False,
    )
    base_rate = master_df["dust_event_next_day"].mean()
    assert res["mean_ap"] > 2 * base_rate          # real ranking skill
    assert res["mean_roc"] > 0.7
    # Decision thresholds are tuned per fold, not a fixed default.
    assert all(0.0 < t < 1.0 for t in res["fold_threshold"])


def test_ablation_recovers_known_drivers(master_df, tmp_path):
    """The ablation surfaces the satellite/surface drivers built into the data."""
    # Compact feature subset → groups: wind_speed, humidity_dryness, vegetation,
    # wind_direction, albedo, soil_texture.
    feats = [
        "ws_max", "rh_mean", "ndvi", "wind_n_mean",
        "albedo_anomaly", "soil_clay_0-5cm",
    ]
    table = run_group_ablation(
        master_df, feats, n_splits=4, xgb_params={"n_estimators": 80},
        n_bootstrap=300, output_dir=tmp_path,
    )
    ranked = table.sort_values("incremental_pr_auc", ascending=False)
    # A known driver (vegetation or wind direction) tops the ranking, above the
    # pure-noise soil-texture group.
    assert ranked.iloc[0]["group"] in {"vegetation", "wind_direction"}
    soil = table.set_index("group").loc["soil_texture", "incremental_pr_auc"]
    veg = table.set_index("group").loc["vegetation", "incremental_pr_auc"]
    assert veg > soil
