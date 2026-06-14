"""
Feature engineering: albedo anomaly, ERA5 lags, temporal encodings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ERA5_LAG_COLS = [
    "ws_max",
    "ws_mean",
    "blh_min",
    "rh_mean",
    "ustar_max",
    "sm_mean",
    "t2m_mean",
]


def compute_albedo_anomaly(
    albedo_df: pd.DataFrame,
    baseline_years: list[int] | None = None,
    study_years: list[int] | None = None,
    doy_window: int = 15,
) -> pd.DataFrame:
    """
    Compute shortwave broadband albedo anomaly for the study period.

    anomaly(t) = albedo(t) - baseline_mean(DOY +/- doy_window)
    using baseline years (default 2015-2017).
    """
    baseline_years = baseline_years or [2015, 2016, 2017]
    study_years = study_years or [2018, 2019, 2020]

    df = albedo_df.copy()
    if "albedo_wsb" not in df.columns:
        raise ValueError("albedo_df must contain column 'albedo_wsb'")

    df["doy"] = getattr(df.index, "dayofyear", df.index.day_of_year)

    baseline = df[df.index.year.isin(baseline_years)].copy()
    study = df[df.index.year.isin(study_years)].copy()

    def doy_distance(a: np.ndarray, b: int) -> np.ndarray:
        d = np.abs(a - b)
        return np.minimum(d, 366 - d)

    baseline_means: dict[int, float] = {}
    for doy_target in study["doy"].unique():
        mask = doy_distance(baseline["doy"].values, doy_target) <= doy_window
        if mask.sum() > 0:
            baseline_means[doy_target] = baseline.loc[mask, "albedo_wsb"].mean()
        else:
            baseline_means[doy_target] = np.nan

    study = study.copy()
    study["albedo_baseline"] = study["doy"].map(baseline_means)
    study["albedo_anomaly"] = study["albedo_wsb"] - study["albedo_baseline"]

    study["albedo_anom_3d"] = (
        study["albedo_anomaly"].rolling(3, min_periods=1).mean()
    )
    study["albedo_anom_7d"] = (
        study["albedo_anomaly"].rolling(7, min_periods=1).mean()
    )

    return study.drop(columns=["doy"])


def add_lag_features(
    df: pd.DataFrame,
    cols: list[str],
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Add lagged copies and first-differences of selected columns."""
    lags = lags or [1, 2, 3]
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        df[f"{col}_diff1"] = df[col].diff(1)
        df[f"{col}_diff2"] = df[col].diff(2)
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode day-of-year and month as sine/cosine pairs."""
    df = df.copy()
    doy = getattr(df.index, "dayofyear", df.index.day_of_year)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    month = df.index.month.values
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    return df


# The model's feature set is FULL_FEATURES below. It is assembled from a
# meteorology/soil/vegetation block plus the satellite reflectivity (albedo)
# features, which the driver ablation then evaluates group by group.
BASELINE_FEATURES = [
    "ws_max",
    "ws_mean",
    "blh_min",
    "blh_mean",
    "rh_mean",
    "t2m_mean",
    "sp_mean",
    "tcwv_mean",
    "sm_mean",
    "soilt_mean",
    "ustar_max",
    "wind_n_mean",
    "wind_e_mean",
    "northerly_frac",
    "precip_sum",
    "precip_7d",
    *[f"{c}_lag{l}" for c in ERA5_LAG_COLS for l in [1, 2, 3]],
    *[f"{c}_diff{d}" for c in ERA5_LAG_COLS for d in [1, 2]],
    "ndvi",
    "soil_clay_0-5cm",
    "soil_sand_0-5cm",
    "soil_silt_0-5cm",
    "soil_ocs_0-5cm",
    "soil_bdod_0-5cm",
    "doy_sin",
    "doy_cos",
    "month_sin",
    "month_cos",
]

ALBEDO_FEATURES = [
    "albedo_wsb",
    "albedo_anomaly",
    "albedo_anom_3d",
    "albedo_anom_7d",
]

FULL_FEATURES = BASELINE_FEATURES + ALBEDO_FEATURES


# ---------------------------------------------------------------------------
# Feature sets for the keyless real-data path (Open-Meteo ERA5 + ORNL MODIS).
# Open-Meteo's archive does not expose boundary-layer height, friction velocity
# or column water vapour, so the real baseline uses the available daily fields
# (plus gust, VPD and cloud cover, which are strong dust proxies) instead.
# ---------------------------------------------------------------------------

REAL_LAG_COLS = ["ws_max", "ws_mean", "rh_mean", "sm_mean", "t2m_mean"]

REAL_BASELINE_FEATURES = [
    "ws_max",
    "ws_mean",
    "gust_max",
    "rh_mean",
    "rh_min",
    "t2m_mean",
    "t2m_max",
    "td_mean",
    "sp_mean",
    "sm_mean",
    "soilt_mean",
    "cloud_mean",
    "vpd_mean",
    "wind_n_mean",
    "wind_e_mean",
    "northerly_frac",
    "precip_sum",
    "precip_7d",
    *[f"{c}_lag{l}" for c in REAL_LAG_COLS for l in [1, 2, 3]],
    *[f"{c}_diff{d}" for c in REAL_LAG_COLS for d in [1, 2]],
    "ndvi",
    "soil_clay_0-5cm",
    "soil_sand_0-5cm",
    "soil_silt_0-5cm",
    "soil_ocs_0-5cm",
    "soil_bdod_0-5cm",
    "doy_sin",
    "doy_cos",
    "month_sin",
    "month_cos",
]

REAL_FULL_FEATURES = REAL_BASELINE_FEATURES + ALBEDO_FEATURES


# ---------------------------------------------------------------------------
# Physically-motivated feature groups for the driver-ablation analysis.
# Each group is a candidate "driver" of next-day dust whose incremental value
# we quantify by dropping it and measuring the change in PR-AUC.
# ---------------------------------------------------------------------------

def assign_group(feature: str) -> str:
    """Map a feature name to its physical driver group."""
    f = feature
    if "albedo" in f:
        return "albedo"
    if f.startswith(("wind_n", "wind_e", "northerly", "wdir")):
        return "wind_direction"
    if f.startswith(("ws_", "gust", "ustar")):
        return "wind_speed"
    if f.startswith(("rh_", "td_", "vpd", "tcwv", "cloud")):
        return "humidity_dryness"
    if f.startswith(("sm_", "precip")):
        return "antecedent_moisture"
    if f.startswith(("t2m", "soilt", "blh")):
        return "thermal_blh"
    if f.startswith("sp"):
        return "pressure"
    if f.startswith("ndvi"):
        return "vegetation"
    if f.startswith("soil_"):
        return "soil_texture"
    if f.startswith(("doy_", "month_")):
        return "seasonality"
    return "other"


def build_feature_groups(features: list[str]) -> dict[str, list[str]]:
    """Bucket a feature list into ordered driver groups (non-empty only)."""
    groups: dict[str, list[str]] = {}
    for f in features:
        groups.setdefault(assign_group(f), []).append(f)
    return groups
