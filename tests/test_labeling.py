"""Tests for dust-event labeling and master dataset construction."""

import numpy as np
import pandas as pd

from src.acquisition import daily_visibility_flag
from src.labeling import (
    build_full_dataset,
    build_labels,
    build_master_dataframe,
    build_nddi_daily,
)


class TestVisibilityFlag:
    def test_flag_true_only_below_threshold(self):
        idx = pd.date_range("2018-01-01", periods=48, freq="h")
        vis = pd.DataFrame({"visibility_m": [5000.0] * 48}, index=idx)
        vis.iloc[3, 0] = 800.0  # one dusty hour on day 1
        flag = daily_visibility_flag(vis, threshold_m=1000.0)
        assert bool(flag.loc["2018-01-01"]) is True
        assert bool(flag.loc["2018-01-02"]) is False


class TestNddiDaily:
    def test_forward_fill_limited_to_composite_period(self):
        dates = pd.DatetimeIndex(["2018-01-01", "2018-01-17"])  # 16-day gap
        mod09 = pd.DataFrame({"nddi": [0.1, 0.2]}, index=dates)
        daily = build_nddi_daily(mod09)
        assert daily.loc["2018-01-08"] == 0.1  # within 7-day ffill window
        assert np.isnan(daily.loc["2018-01-09"])  # beyond limit -> NaN


class TestBuildLabels:
    def test_dual_criterion(self):
        idx = pd.date_range("2018-01-01", periods=4, freq="D")
        nddi = pd.Series([0.1, 0.1, -0.1, -0.1], index=idx)
        vis = pd.Series([True, False, True, False], index=idx)
        labels = build_labels(nddi, vis)
        # Event only when NDDI > 0 AND visibility flag is True
        assert labels.tolist() == [1, 0, 0, 0]

    def test_missing_visibility_treated_as_no_event(self):
        idx = pd.date_range("2018-01-01", periods=2, freq="D")
        nddi = pd.Series([0.1, 0.1], index=idx)
        vis = pd.Series([True], index=idx[:1])  # day 2 missing
        labels = build_labels(nddi, vis)
        assert labels.tolist() == [1, 0]


def _minimal_station_inputs(start="2017-12-25", end="2018-01-31"):
    idx = pd.date_range(start, end, freq="D")
    n = len(idx)
    rng = np.random.default_rng(0)
    era5 = pd.DataFrame(
        {
            c: rng.random(n)
            for c in [
                "ws_max", "ws_mean", "blh_min", "blh_mean", "rh_mean",
                "t2m_mean", "sp_mean", "tcwv_mean", "sm_mean", "soilt_mean",
                "ustar_max", "precip_sum", "precip_7d",
            ]
        },
        index=idx,
    )
    albedo = pd.DataFrame(
        {
            "albedo_wsb": rng.random(n),
            "albedo_anomaly": rng.random(n),
            "albedo_anom_3d": rng.random(n),
            "albedo_anom_7d": rng.random(n),
        },
        index=idx,
    )
    mod09 = pd.DataFrame(
        {"ndvi": 0.1, "nddi": 0.1},
        index=pd.date_range(start, end, freq="8D"),
    )
    vis_flag = pd.Series(False, index=idx, name="vis_dust_flag")
    vis_flag.loc["2018-01-15"] = True  # one dust day
    soil = {"soil_clay_0-5cm": 50.0, "soil_sand_0-5cm": 800.0}
    return era5, albedo, mod09, vis_flag, soil


class TestMasterDataframe:
    def test_label_is_next_day_event(self):
        era5, albedo, mod09, vis_flag, soil = _minimal_station_inputs()
        df = build_master_dataframe(
            "riyadh", era5, albedo, mod09, vis_flag, soil,
            study_start="2018", study_end="2018",
        )
        # Dust day is Jan 15 -> features of Jan 14 carry the positive label
        assert df.loc["2018-01-14", "dust_event_next_day"] == 1
        assert df.loc["2018-01-15", "dust_event_next_day"] == 0

    def test_restricted_to_study_period(self):
        era5, albedo, mod09, vis_flag, soil = _minimal_station_inputs()
        df = build_master_dataframe(
            "riyadh", era5, albedo, mod09, vis_flag, soil,
            study_start="2018", study_end="2018",
        )
        assert df.index.min() >= pd.Timestamp("2018-01-01")

    def test_no_nan_labels(self):
        era5, albedo, mod09, vis_flag, soil = _minimal_station_inputs()
        df = build_master_dataframe(
            "riyadh", era5, albedo, mod09, vis_flag, soil,
            study_start="2018", study_end="2018",
        )
        assert df["dust_event_next_day"].notna().all()

    def test_static_soil_broadcast(self):
        era5, albedo, mod09, vis_flag, soil = _minimal_station_inputs()
        df = build_master_dataframe(
            "riyadh", era5, albedo, mod09, vis_flag, soil,
            study_start="2018", study_end="2018",
        )
        assert (df["soil_clay_0-5cm"] == 50.0).all()


class TestBuildFullDataset:
    def test_concat_and_sort(self):
        era5, albedo, mod09, vis_flag, soil = _minimal_station_inputs()
        df_a = build_master_dataframe(
            "a_station", era5, albedo, mod09, vis_flag, soil,
            study_start="2018", study_end="2018",
        )
        df_b = df_a.copy()
        df_b["station"] = "b_station"
        combined = build_full_dataset([df_a, df_b])
        assert "date" in combined.columns
        assert len(combined) == len(df_a) + len(df_b)
        assert combined["station"].nunique() == 2
