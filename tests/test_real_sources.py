"""Offline tests for keyless real-source parsing (network mocked)."""


import numpy as np
import pandas as pd

import src.real_sources as rs
from src.labeling import build_visibility_labels


class _FakeResp:
    def __init__(self, json_data=None, text=None):
        self._json = json_data
        self.text = text
        self.status_code = 200

    def json(self):
        return self._json

    def raise_for_status(self):
        pass


class TestOpenMeteo:
    def test_daily_aggregation(self, monkeypatch):
        times = pd.date_range("2020-06-01", periods=48, freq="h")
        hourly = {
            "time": [t.isoformat() for t in times],
            "temperature_2m": [30.0] * 24 + [40.0] * 24,
            "dew_point_2m": [5.0] * 48,
            "relative_humidity_2m": [20.0] * 48,
            "surface_pressure": [950.0] * 48,
            "wind_speed_10m": [2.0] * 23 + [12.0] + [3.0] * 24,
            "wind_gusts_10m": [5.0] * 48,
            "wind_direction_10m": [0.0] * 24 + [90.0] * 24,  # day1 from N, day2 from E
            "precipitation": [0.1] * 48,
            "soil_temperature_0_to_7cm": [28.0] * 48,
            "soil_moisture_0_to_7cm": [0.05] * 48,
            "cloud_cover": [10.0] * 48,
        }
        monkeypatch.setattr(
            rs, "_get", lambda *a, **k: _FakeResp(json_data={"hourly": hourly})
        )
        out = rs.fetch_openmeteo_daily(24.9, 46.7, "2020-06-01", "2020-06-02")
        assert len(out) == 2
        assert out["ws_max"].iloc[0] == 12.0  # peak gust-day captured
        assert np.isclose(out["precip_sum"].iloc[0], 0.1 * 24, atol=1e-6)
        assert (out["vpd_mean"] > 0).all()  # dry desert air
        assert out["t2m_max"].iloc[1] == 40.0
        # Day 1 wind entirely from N -> northerly_frac 1; day 2 from E -> 0
        assert np.isclose(out["northerly_frac"].iloc[0], 1.0)
        assert np.isclose(out["northerly_frac"].iloc[1], 0.0)
        # Northerly component positive on day 1, ~0 on day 2 (easterly)
        assert out["wind_n_mean"].iloc[0] > out["wind_n_mean"].iloc[1]


class TestISD:
    def test_visibility_parse_and_missing(self, monkeypatch):
        csv = (
            "DATE,VIS\n"
            "2020-01-01T00:00:00,010000,1,9,9\n"
            "2020-01-01T06:00:00,000800,1,9,9\n"   # dust hour
            "2020-01-02T00:00:00,999999,9,9,9\n"   # missing -> NaN
        )
        monkeypatch.setattr(rs, "_get", lambda *a, **k: _FakeResp(text=csv))
        df = rs.fetch_isd_visibility("404370", "99999", 2020)
        assert df["visibility_m"].iloc[0] == 10000
        assert df["visibility_m"].iloc[1] == 800
        assert np.isnan(df["visibility_m"].iloc[2])

    def test_daily_flag_from_hours(self, monkeypatch):
        csv = (
            "DATE,VIS\n"
            "2020-01-01T00:00:00,010000,1,9,9\n"
            "2020-01-01T06:00:00,000800,1,9,9\n"
            "2020-01-02T00:00:00,005000,1,9,9\n"
        )
        monkeypatch.setattr(rs, "_get", lambda *a, **k: _FakeResp(text=csv))
        flag = rs.fetch_visibility_flag("404370", "99999", [2020], threshold_m=1000)
        assert bool(flag.loc["2020-01-01"]) is True
        assert bool(flag.loc["2020-01-02"]) is False


class TestMODIS:
    def test_indices_and_liang_albedo(self, monkeypatch):
        dates = pd.DataFrame(
            {
                "modis_date": ["A2020153", "A2020161"],
                "calendar_date": pd.to_datetime(["2020-06-01", "2020-06-09"]),
            }
        )
        monkeypatch.setattr(rs, "_ornl_dates", lambda lat, lon, product="MOD09A1": dates)

        # Constant reflectance per band so the albedo is analytic.
        refl = {f"sur_refl_b0{i}": 0.30 for i in range(1, 8)}
        refl["sur_refl_b01"] = 0.20  # red
        refl["sur_refl_b02"] = 0.40  # nir
        refl["sur_refl_b06"] = 0.35  # swir

        monkeypatch.setattr(
            rs,
            "_ornl_band_mean",
            lambda lat, lon, band, md, km, product: refl[band],
        )

        df = rs.fetch_mod09a1_indices(
            24.9, 46.7, "2020-06-01", "2020-06-30", km=10, sleep_s=0, progress=False
        )
        assert len(df) == 2
        # NDVI = (nir - red)/(nir + red) = (0.4-0.2)/(0.6)
        assert np.isclose(df["ndvi"].iloc[0], 0.2 / 0.6)
        # Liang shortwave albedo with the mocked reflectances
        expected = (
            rs.LIANG_INTERCEPT
            + 0.160 * 0.20 + 0.291 * 0.40 + 0.243 * 0.30
            + 0.116 * 0.30 + 0.112 * 0.30 + 0.081 * 0.30
        )
        assert np.isclose(df["albedo_wsb"].iloc[0], expected)

    def test_band_mean_masks_fill_values(self, monkeypatch):
        payload = {
            "scale": 0.0001,
            "subset": [{"data": [3000, 4000, -28672, 5000]}],  # one fill value
        }
        monkeypatch.setattr(rs, "_get", lambda *a, **k: _FakeResp(json_data=payload))
        val = rs._ornl_band_mean(24.9, 46.7, "sur_refl_b01", "A2020153", 10, "MOD09A1")
        # Mean of 0.30, 0.40, 0.50 (fill masked out)
        assert np.isclose(val, 0.40)


class TestVisibilityLabel:
    def test_label_reindex_and_fill(self):
        vis = pd.Series(
            [True, False], index=pd.to_datetime(["2020-01-01", "2020-01-03"])
        )
        idx = pd.date_range("2020-01-01", "2020-01-03", freq="D")
        label = build_visibility_labels(vis, idx)
        assert label.tolist() == [1, 0, 0]  # missing 01-02 filled False
