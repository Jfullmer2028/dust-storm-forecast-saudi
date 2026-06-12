"""End-to-end smoke test on the bundled synthetic data (no network)."""

from pathlib import Path

import numpy as np
import pytest

from src.acquisition import load_config, load_synthetic_station_bundle
from src.evaluation import full_statistical_analysis
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
                study_end="2020",
            )
        )
    return build_full_dataset(station_dfs)


def test_master_dataset_shape_and_labels(master_df):
    assert len(master_df) > 3000
    assert master_df["station"].nunique() == 3
    positives = master_df["dust_event_next_day"].sum()
    assert 0 < positives < len(master_df) * 0.5  # imbalanced but non-empty
    assert set(FULL_FEATURES) <= set(master_df.columns)


def test_cv_runs_and_albedo_improves_f2(master_df, tmp_path):
    fast = {"n_estimators": 60}
    baseline = run_cross_validation(
        master_df, BASELINE_FEATURES, n_splits=3, xgb_params=fast, verbose=False
    )
    full = run_cross_validation(
        master_df, FULL_FEATURES, n_splits=3, xgb_params=fast, verbose=False
    )
    # Synthetic data plants an albedo precursor signal, so the full model
    # must beat the baseline by a wide margin if the pipeline is wired right.
    assert full["mean_f2"] > baseline["mean_f2"]

    stats = full_statistical_analysis(
        baseline, full, n_bootstrap=200, output_dir=tmp_path
    )
    assert np.isfinite(stats["wilcoxon"]["p"])
    assert stats["bootstrap"]["lo"] <= stats["bootstrap"]["hi"]
