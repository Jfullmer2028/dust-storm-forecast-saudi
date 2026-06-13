"""Tests for feature engineering: albedo anomaly, lags, temporal encodings."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    ALBEDO_FEATURES,
    BASELINE_FEATURES,
    FULL_FEATURES,
    REAL_FULL_FEATURES,
    add_lag_features,
    add_temporal_features,
    assign_group,
    build_feature_groups,
    compute_albedo_anomaly,
)


def _make_albedo_df(constant_baseline=0.3, study_value=0.35):
    """Albedo series: constant in baseline years, shifted in study years."""
    dates = pd.date_range("2015-01-01", "2020-12-31", freq="D")
    values = np.where(dates.year <= 2017, constant_baseline, study_value)
    df = pd.DataFrame({"albedo_wsb": values}, index=dates)
    df.index.name = "date"
    return df


class TestComputeAlbedoAnomaly:
    def test_anomaly_is_difference_from_baseline(self):
        df = _make_albedo_df(constant_baseline=0.30, study_value=0.35)
        out = compute_albedo_anomaly(df)
        # Baseline is constant 0.30 so anomaly should be ~0.05 everywhere
        assert np.allclose(out["albedo_anomaly"].dropna(), 0.05, atol=1e-9)

    def test_output_restricted_to_study_years(self):
        df = _make_albedo_df()
        out = compute_albedo_anomaly(df)
        assert set(out.index.year.unique()) <= {2018, 2019, 2020}

    def test_zero_anomaly_when_study_equals_baseline(self):
        df = _make_albedo_df(constant_baseline=0.30, study_value=0.30)
        out = compute_albedo_anomaly(df)
        assert np.allclose(out["albedo_anomaly"].dropna(), 0.0, atol=1e-9)

    def test_rolling_smoothed_columns_present(self):
        out = compute_albedo_anomaly(_make_albedo_df())
        assert {"albedo_anom_3d", "albedo_anom_7d"} <= set(out.columns)

    def test_missing_column_raises(self):
        bad = pd.DataFrame(
            {"x": [1.0]}, index=pd.DatetimeIndex(["2018-01-01"])
        )
        with pytest.raises(ValueError, match="albedo_wsb"):
            compute_albedo_anomaly(bad)

    def test_doy_window_wraps_around_year_end(self):
        # Baseline values only near Jan 1; a Dec 31 study date must still
        # find baseline observations through the circular DOY distance.
        baseline_dates = pd.date_range("2015-01-01", "2015-01-10", freq="D")
        study_dates = pd.DatetimeIndex(["2018-12-31"])
        df = pd.DataFrame(
            {"albedo_wsb": [0.3] * len(baseline_dates) + [0.4]},
            index=baseline_dates.append(study_dates),
        )
        out = compute_albedo_anomaly(df, doy_window=15)
        assert np.isclose(out.loc["2018-12-31", "albedo_anomaly"], 0.1)


class TestLagFeatures:
    def test_lag_values_shifted(self):
        idx = pd.date_range("2018-01-01", periods=5, freq="D")
        df = pd.DataFrame({"ws_max": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
        out = add_lag_features(df, ["ws_max"], lags=[1, 2])
        assert np.isnan(out["ws_max_lag1"].iloc[0])
        assert out["ws_max_lag1"].iloc[1] == 1.0
        assert out["ws_max_lag2"].iloc[2] == 1.0
        assert out["ws_max_diff1"].iloc[1] == 1.0

    def test_missing_columns_skipped(self):
        idx = pd.date_range("2018-01-01", periods=3, freq="D")
        df = pd.DataFrame({"a": [1, 2, 3]}, index=idx)
        out = add_lag_features(df, ["not_there"])
        assert list(out.columns) == ["a"]


class TestTemporalFeatures:
    def test_cyclical_encodings_bounded(self):
        idx = pd.date_range("2018-01-01", "2018-12-31", freq="D")
        out = add_temporal_features(pd.DataFrame(index=idx))
        for col in ["doy_sin", "doy_cos", "month_sin", "month_cos"]:
            assert out[col].between(-1, 1).all()

    def test_continuity_across_year_boundary(self):
        idx = pd.DatetimeIndex(["2018-12-31", "2019-01-01"])
        out = add_temporal_features(pd.DataFrame(index=idx))
        # Dec 31 and Jan 1 must be close in cyclical space
        assert abs(out["doy_sin"].iloc[0] - out["doy_sin"].iloc[1]) < 0.05


class TestFeatureSets:
    def test_full_is_baseline_plus_albedo(self):
        assert FULL_FEATURES == BASELINE_FEATURES + ALBEDO_FEATURES

    def test_baseline_has_no_albedo_features(self):
        assert not any("albedo" in f for f in BASELINE_FEATURES)

    def test_no_duplicate_features(self):
        assert len(FULL_FEATURES) == len(set(FULL_FEATURES))


class TestFeatureGroups:
    def test_assignments(self):
        cases = {
            "albedo_anomaly": "albedo",
            "wind_n_mean": "wind_direction",
            "northerly_frac": "wind_direction",
            "ws_max": "wind_speed",
            "ws_max_lag1": "wind_speed",
            "gust_max": "wind_speed",
            "rh_mean": "humidity_dryness",
            "vpd_mean": "humidity_dryness",
            "sm_mean": "antecedent_moisture",
            "precip_7d": "antecedent_moisture",
            "t2m_mean": "thermal_blh",
            "sp_mean": "pressure",
            "ndvi": "vegetation",
            "soil_clay_0-5cm": "soil_texture",
            "doy_sin": "seasonality",
        }
        for feat, grp in cases.items():
            assert assign_group(feat) == grp, feat

    def test_every_full_feature_grouped_no_other(self):
        for feats in (FULL_FEATURES, REAL_FULL_FEATURES):
            groups = build_feature_groups(feats)
            assert "other" not in groups  # every feature has a physical group
            # Partition is exact and lossless
            assert sum(len(v) for v in groups.values()) == len(feats)

    def test_wind_direction_group_present(self):
        groups = build_feature_groups(FULL_FEATURES)
        assert set(groups["wind_direction"]) == {
            "wind_n_mean", "wind_e_mean", "northerly_frac"
        }
