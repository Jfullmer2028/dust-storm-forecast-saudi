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


def test_baseline_is_not_degenerate(master_df):
    """The meteorological baseline must have real 24h skill (no 0.000 folds)."""
    baseline = run_cross_validation(
        master_df,
        BASELINE_FEATURES,
        n_splits=5,
        xgb_params={"n_estimators": 120},
        verbose=False,
    )
    # Synoptic persistence gives the baseline genuine next-day signal.
    assert baseline["mean_f2"] > 0.20
    assert (baseline["fold_f2"] > 0.0).all()


def test_cv_runs_and_albedo_improves_f2(master_df, tmp_path):
    fast = {"n_estimators": 120}
    baseline = run_cross_validation(
        master_df, BASELINE_FEATURES, n_splits=6, xgb_params=fast, verbose=False
    )
    full = run_cross_validation(
        master_df, FULL_FEATURES, n_splits=6, xgb_params=fast, verbose=False
    )
    # Albedo carries an incremental precursor signal: a modest but consistent
    # improvement, not an artefactual landslide.
    assert full["mean_f2"] > baseline["mean_f2"]
    assert 0.0 < (full["mean_f2"] - baseline["mean_f2"]) < 0.5

    # Decision thresholds are tuned, never the naive 0.5 default everywhere.
    assert all(0.0 < t < 1.0 for t in full["fold_threshold"])

    stats = full_statistical_analysis(
        baseline, full, n_bootstrap=300, output_dir=tmp_path
    )
    assert np.isfinite(stats["wilcoxon"]["p"])
    assert stats["bootstrap"]["lo"] <= stats["bootstrap"]["hi"]
