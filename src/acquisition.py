"""
Data acquisition from NOAA ISD, MODIS (GEE), ERA5 (CDS), and SoilGrids.

Synthetic mode generates representative CSVs locally so the pipeline runs
without external API keys or accounts.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yaml

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_stations(config: dict) -> dict[str, dict[str, float | str]]:
    return config["stations"]


# ---------------------------------------------------------------------------
# NOAA ISD visibility (real data)
# ---------------------------------------------------------------------------

def download_isd_visibility(
    wmo_id: str,
    wban: str,
    year: int,
) -> pd.DataFrame:
    """
    Parse visibility from full NOAA ISD files via the `isd` package.

    Downloads the gzipped ISD archive, parses hourly records, and returns
    visibility in metres. Missing/sentinel values are stored as NaN.
    """
    import gzip

    from isd import Batch

    url = (
        f"https://www.ncei.noaa.gov/pub/data/noaa/{year}/"
        f"{wmo_id}-{wban}-{year}.gz"
    )
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    text = gzip.decompress(resp.content).decode("utf-8", errors="replace")
    batch = Batch.parse(text)
    records = batch.to_dict()

    if not records:
        raise ValueError(f"No ISD records parsed for {wmo_id} ({year})")

    df = pd.DataFrame(records)[["datetime", "visibility"]]
    df = df.rename(columns={"visibility": "visibility_m"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()
    df["visibility_m"] = pd.to_numeric(df["visibility_m"], errors="coerce")
    df.loc[df["visibility_m"] < 0, "visibility_m"] = float("nan")
    df.index.name = "datetime"
    return df


def daily_visibility_flag(
    vis_df: pd.DataFrame,
    threshold_m: float = 1000.0,
) -> pd.Series:
    """True on any day with at least one hourly visibility observation <= threshold."""
    below = vis_df["visibility_m"] <= threshold_m
    daily = below.groupby(below.index.normalize()).max()
    daily.index = pd.to_datetime(daily.index)
    return daily.rename("vis_dust_flag")


# ---------------------------------------------------------------------------
# MODIS MCD43A3 albedo via Google Earth Engine (real data)
# ---------------------------------------------------------------------------

def get_albedo_timeseries(
    station_name: str,
    stations: dict,
    radius_m: int = 200_000,
    start: str = "2015-01-01",
    end: str = "2020-12-31",
) -> pd.DataFrame:
    """
    Extract mean shortwave broadband white-sky albedo (MCD43A3) within radius.
    """
    import ee

    ee.Initialize()

    coords = stations[station_name]
    point = ee.Geometry.Point([coords["lon"], coords["lat"]])
    roi = point.buffer(radius_m)

    collection = (
        ee.ImageCollection("MODIS/061/MCD43A3")
        .filterDate(start, end)
        .filterBounds(roi)
        .select("Albedo_WSA_shortwave")
    )

    def reduce_image(image):
        stats = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=500,
            maxPixels=int(1e9),
            bestEffort=True,
        )
        val = ee.Number(stats.get("Albedo_WSA_shortwave")).multiply(0.001)
        return (
            image.set("date", image.date().format("YYYY-MM-dd"))
            .set("albedo_wsb", val)
        )

    reduced = collection.map(reduce_image)
    info = reduced.aggregate_array("albedo_wsb").getInfo()
    dates = reduced.aggregate_array("date").getInfo()

    df = pd.DataFrame({"date": pd.to_datetime(dates), "albedo_wsb": info})
    df = df.set_index("date").sort_index()
    df["station"] = station_name
    return df


# ---------------------------------------------------------------------------
# MOD09A1 NDVI / NDDI via GEE (real data)
# ---------------------------------------------------------------------------

def get_mod09a1_timeseries(
    station_name: str,
    stations: dict,
    radius_m: int = 200_000,
    start: str = "2018-01-01",
    end: str = "2020-12-31",
) -> pd.DataFrame:
    """Extract mean NDVI and NDDI from MOD09A1 8-day composites within radius."""
    import ee

    ee.Initialize()

    coords = stations[station_name]
    point = ee.Geometry.Point([coords["lon"], coords["lat"]])
    roi = point.buffer(radius_m)

    collection = (
        ee.ImageCollection("MODIS/061/MOD09A1")
        .filterDate(start, end)
        .filterBounds(roi)
        .select(["sur_refl_b01", "sur_refl_b02", "sur_refl_b06"])
    )

    def compute_indices(image):
        red = image.select("sur_refl_b01").multiply(0.0001)
        nir = image.select("sur_refl_b02").multiply(0.0001)
        swir = image.select("sur_refl_b06").multiply(0.0001)

        ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
        ndwi = nir.subtract(swir).divide(nir.add(swir)).rename("NDWI")
        nddi = ndvi.subtract(ndwi).divide(ndvi.add(ndwi)).rename("NDDI")

        stats = ndvi.addBands(nddi).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=500,
            maxPixels=int(1e9),
            bestEffort=True,
        )
        return (
            image.set("date", image.date().format("YYYY-MM-dd"))
            .set("ndvi_mean", stats.get("NDVI"))
            .set("nddi_mean", stats.get("NDDI"))
        )

    reduced = collection.map(compute_indices)
    dates = reduced.aggregate_array("date").getInfo()
    ndvi = reduced.aggregate_array("ndvi_mean").getInfo()
    nddi = reduced.aggregate_array("nddi_mean").getInfo()

    df = pd.DataFrame(
        {"date": pd.to_datetime(dates), "ndvi": ndvi, "nddi": nddi}
    )
    df = df.set_index("date").sort_index()
    df["station"] = station_name
    return df


# ---------------------------------------------------------------------------
# ERA5 via CDS API (real data)
# ---------------------------------------------------------------------------

ERA5_VARIABLE_MAP = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "surface_pressure": "sp",
    "total_column_water_vapour": "tcwv",
    "boundary_layer_height": "blh",
    "soil_temperature_level_1": "stl1",
    "volumetric_soil_water_layer_1": "swvl1",
    "friction_velocity": "zust",
    "total_precipitation": "tp",
}


def download_era5(
    lat: float,
    lon: float,
    years: list[int],
    output_nc: str | Path,
    variables: list[str] | None = None,
) -> None:
    """Download ERA5 hourly reanalysis for a +/-2 degree bounding box."""
    import cdsapi

    variables = variables or list(ERA5_VARIABLE_MAP.keys())
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": variables,
            "year": [str(y) for y in years],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(0, 24)],
            "area": [lat + 2, lon - 2, lat - 2, lon + 2],
            "format": "netcdf",
        },
        str(output_nc),
    )


def era5_to_daily_features(
    nc_path: str | Path,
    lat: float,
    lon: float,
) -> pd.DataFrame:
    """
    Load ERA5 NetCDF, extract nearest grid point, compute daily aggregates
    for the 24-hour window ending at each calendar day.
    """
    import xarray as xr

    ds = xr.open_dataset(nc_path)
    ds_pt = ds.sel(latitude=lat, longitude=lon, method="nearest")

    df = ds_pt.to_dataframe().reset_index()
    time_col = "time" if "time" in df.columns else "valid_time"
    df = df.rename(columns={time_col: "datetime"})
    df = df.set_index("datetime").sort_index()

    # Map any short ERA5 variable codes back to their canonical names.
    inv = {v: k for k, v in ERA5_VARIABLE_MAP.items()}
    for col in df.columns:
        if col in inv:
            df = df.rename(columns={col: inv[col]})

    u_col = "u10" if "u10" in df.columns else "10m_u_component_of_wind"
    v_col = "v10" if "v10" in df.columns else "10m_v_component_of_wind"
    df["wind_speed_10m"] = np.sqrt(df[u_col] ** 2 + df[v_col] ** 2)

    t2m = "t2m" if "t2m" in df.columns else "2m_temperature"
    d2m = "d2m" if "d2m" in df.columns else "2m_dewpoint_temperature"
    T = df[t2m] - 273.15
    Td = df[d2m] - 273.15
    df["rh2m"] = (
        100
        * np.exp((17.625 * Td) / (243.04 + Td))
        / np.exp((17.625 * T) / (243.04 + T))
    )

    blh = "blh" if "blh" in df.columns else "boundary_layer_height"
    sp = "sp" if "sp" in df.columns else "surface_pressure"
    tcwv = "tcwv" if "tcwv" in df.columns else "total_column_water_vapour"
    swvl1 = "swvl1" if "swvl1" in df.columns else "volumetric_soil_water_layer_1"
    stl1 = "stl1" if "stl1" in df.columns else "soil_temperature_level_1"
    zust = "zust" if "zust" in df.columns else "friction_velocity"
    tp = "tp" if "tp" in df.columns else "total_precipitation"

    agg = df.resample("D").agg(
        ws_max=("wind_speed_10m", "max"),
        ws_mean=("wind_speed_10m", "mean"),
        blh_min=(blh, "min"),
        blh_mean=(blh, "mean"),
        rh_mean=("rh2m", "mean"),
        t2m_mean=(t2m, "mean"),
        sp_mean=(sp, "mean"),
        tcwv_mean=(tcwv, "mean"),
        sm_mean=(swvl1, "mean"),
        soilt_mean=(stl1, "mean"),
        ustar_max=(zust, "max"),
        precip_sum=(tp, "sum"),
    )
    agg["precip_7d"] = (
        agg["precip_sum"].rolling(7, min_periods=1).sum().shift(1)
    )
    return agg


# ---------------------------------------------------------------------------
# SoilGrids REST API (real data)
# ---------------------------------------------------------------------------

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"


def fetch_soilgrids(
    lat: float,
    lon: float,
    properties: list[str] | None = None,
    depths: list[str] | None = None,
) -> dict[str, float]:
    """Fetch SoilGrids v2 properties at a point."""
    properties = properties or ["clay", "sand", "silt", "ocs", "bdod"]
    depths = depths or ["0-5cm"]
    params = {
        "lon": lon,
        "lat": lat,
        "property": properties,
        "depth": depths,
        "value": "mean",
    }
    r = requests.get(SOILGRIDS_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    features: dict[str, float] = {}
    for layer in data["properties"]["layers"]:
        prop = layer["name"]
        for depth_info in layer["depths"]:
            depth_label = depth_info["label"]
            val = depth_info["values"]["mean"]
            features[f"soil_{prop}_{depth_label}"] = val
    return features


def build_soil_dataframe(
    stations: dict,
    properties: list[str] | None = None,
    sleep_s: float = 2.0,
) -> pd.DataFrame:
    """Build static soil DataFrame for all stations (respects rate limits)."""
    rows = []
    for name, coords in stations.items():
        feats = fetch_soilgrids(coords["lat"], coords["lon"], properties)
        feats["station"] = name
        rows.append(feats)
        time.sleep(sleep_s)
    return pd.DataFrame(rows).set_index("station")


# ---------------------------------------------------------------------------
# Synthetic data generation (no API keys required)
# ---------------------------------------------------------------------------

def generate_synthetic_station_data(
    station_name: str,
    lat: float,
    lon: float,
    start: str = "2013-01-01",
    end: str = "2022-12-31",
    seed: int = 42,
    positive_rate: float = 0.10,
    station_idx: int = 0,
) -> dict[str, Any]:
    """
    Generate realistic synthetic per-station data.

    Three latent processes drive the system:

    * A synoptic index ``S_t`` (AR(1), phi=0.75) drives wind, humidity and
      boundary-layer height, so weather *persists* day to day and today's
      conditions carry information about tomorrow's dust.

    * A wind-direction (shamal) index ``D_t`` (AR(1), phi=0.70) raises dust risk
      through northerly flow, independent of wind speed.

    * A slowly-varying surface-state index ``E_t`` (AR(1), phi=0.85) that the
      MODIS **vegetation** signal (NDVI) reflects and that leads dust one day
      ahead — lower green cover indicates a more exposed, erodible surface.

    These give two satellite/surface drivers (vegetation and wind direction)
    with genuine incremental skill over the meteorological fields, providing a
    benchmark on which the driver ablation can be checked. Everything is
    deterministic in ``seed`` + ``station_idx`` (reproducible).
    """
    rng = np.random.default_rng(seed + 1000 * (station_idx + 1))

    dates = pd.date_range(start, end, freq="D")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    n = len(dates)
    doy = np.asarray(getattr(dates, "dayofyear", dates.day_of_year))

    # Per-station heterogeneity (different climates / signal strengths)
    base_rate = rng.uniform(0.07, 0.13)
    met_strength = rng.uniform(0.90, 1.25)
    albedo_gain = rng.uniform(0.90, 1.30)
    erod_weight = rng.uniform(0.65, 1.00)

    # Dust season: Arabian shamal peaks in spring–summer
    season = 0.5 * (1 + np.sin(2 * np.pi * (doy - 100) / 365.25))

    # Latent synoptic index S_t ~ AR(1) — drives weather, persistent
    phi_s = 0.75
    eps = rng.normal(0, 1, n)
    S = np.zeros(n)
    for t in range(1, n):
        S[t] = phi_s * S[t - 1] + np.sqrt(1 - phi_s**2) * eps[t]

    # Latent surface-state index E_t ~ AR(1) — a slowly-varying erodibility /
    # exposure index that the MODIS vegetation signal reflects and that leads
    # dust. High persistence so it survives the 8-day NDVI compositing.
    phi_e = 0.85
    eps_e = rng.normal(0, 1, n)
    E = np.zeros(n)
    for t in range(1, n):
        E[t] = phi_e * E[t - 1] + np.sqrt(1 - phi_e**2) * eps_e[t]

    # Latent wind-direction (shamal) index D_t ~ AR(1) — partly independent of
    # the synoptic strength S, so northerly flow carries dust information beyond
    # wind *speed* alone (the physical basis of the Arabian shamal).
    phi_d = 0.70
    eps_d = rng.normal(0, 1, n)
    D = np.zeros(n)
    for t in range(1, n):
        D[t] = phi_d * D[t - 1] + np.sqrt(1 - phi_d**2) * eps_d[t]

    # --- ERA5-like daily weather driven by the synoptic index ---
    seasonal_ws = 4.0 + 2.5 * np.sin(2 * np.pi * (doy - 100) / 365.25)
    ws_max = np.clip(
        seasonal_ws + 2.6 * met_strength * S + rng.normal(0, 1.3, n), 0.5, 28
    )
    era5 = pd.DataFrame(index=dates)
    era5["ws_max"] = ws_max
    era5["ws_mean"] = ws_max * rng.uniform(0.55, 0.78, n)
    era5["blh_mean"] = np.clip(
        900 + 500 * S + 400 * season + rng.normal(0, 180, n), 150, 3500
    )
    era5["blh_min"] = era5["blh_mean"] * rng.uniform(0.25, 0.50, n)
    era5["rh_mean"] = np.clip(
        32 - 9 * S - 6 * season + rng.normal(0, 6, n), 2, 92
    )
    era5["t2m_mean"] = (
        273.15 + 25 + 15 * np.sin(2 * np.pi * (doy - 180) / 365.25)
        + rng.normal(0, 2, n)
    )
    era5["sp_mean"] = rng.normal(100500, 500, n)
    era5["tcwv_mean"] = np.clip(18 - 6 * season + rng.normal(0, 4, n), 3, 45)
    era5["sm_mean"] = np.clip(
        0.10 - 0.05 * season + rng.normal(0, 0.02, n), 0.01, 0.25
    )
    era5["soilt_mean"] = era5["t2m_mean"] - rng.uniform(2, 8, n)
    era5["ustar_max"] = era5["ws_max"] * 0.08 + np.abs(rng.normal(0, 0.02, n))
    # Wind direction (resultant northerly / easterly components, m/s) and the
    # fraction of the day with northerly flow — driven by S (strength) and D
    # (shamal direction). Northerly component grows with both.
    era5["wind_n_mean"] = 1.4 * S + 1.6 * D + rng.normal(0, 0.8, n)
    era5["wind_e_mean"] = -0.4 * D + rng.normal(0, 1.5, n)
    era5["northerly_frac"] = np.clip(
        0.45 + 0.12 * (S + D) + rng.normal(0, 0.07, n), 0, 1
    )
    precip = rng.exponential(0.0008, n) * (1 - season)
    era5["precip_sum"] = precip
    era5["precip_7d"] = (
        pd.Series(precip, index=dates)
        .rolling(7, min_periods=1)
        .sum()
        .shift(1)
        .fillna(0)
        .values
    )
    era5.index.name = "date"

    # --- Dust intensity on day t: contemporaneous synoptic forcing plus the
    #     surface-state index built the previous day (reflected in NDVI) ---
    E_lag = np.concatenate([[0.0], E[:-1]])
    D_lag = np.concatenate([[0.0], D[:-1]])
    dust_score = (
        1.15 * met_strength * np.clip(S, 0, None)
        + 0.55 * season * np.clip(S, 0, None)
        + 0.95 * erod_weight * np.clip(E_lag, 0, None)
        + 0.60 * np.clip(D_lag, 0, None)  # northerly shamal precursor
        + 0.25 * season
        + rng.normal(0, 0.45, n)
    )
    thr = np.quantile(dust_score, 1 - base_rate)
    dust_events = dust_score > thr

    # --- MODIS albedo: seasonal climatology plus small day-to-day variation ---
    seasonal_albedo = 0.28 + 0.04 * np.sin(2 * np.pi * (doy - 90) / 365.25)
    albedo_wsb = np.clip(
        seasonal_albedo + 0.010 * albedo_gain * rng.normal(0, 1, n) + 0.004,
        0.15,
        0.50,
    )
    albedo_df = pd.DataFrame({"albedo_wsb": albedo_wsb}, index=dates)
    albedo_df.index.name = "date"
    albedo_df["station"] = station_name

    # --- MOD09A1 8-day composites: NDVI carries the surface-state index (lower
    #     green cover where the surface is more exposed/erodible); NDDI spikes
    #     on dust days for the dual-criterion label. ---
    mod09_dates = pd.date_range(start, end, freq="8D")
    e_at_mod = E[np.clip(dates.searchsorted(mod09_dates), 0, n - 1)]
    ndvi_vals = np.clip(
        0.11 - 0.05 * e_at_mod + rng.normal(0, 0.012, len(mod09_dates)), 0.02, 0.45
    )
    nddi_vals = rng.normal(-0.08, 0.10, len(mod09_dates))
    mod09_df = pd.DataFrame(
        {"ndvi": ndvi_vals, "nddi": nddi_vals}, index=mod09_dates
    )
    mod09_df.index.name = "date"
    mod09_df["station"] = station_name

    # --- Hourly visibility: low on dust days ---
    vis_idx = pd.date_range(
        start_ts, end_ts + pd.Timedelta(hours=23), freq="h"
    )
    visibility = rng.uniform(3000, 10000, len(vis_idx))
    vis_df = pd.DataFrame({"visibility_m": visibility}, index=vis_idx)

    nddi_col = mod09_df.columns.get_loc("nddi")
    for i in np.flatnonzero(dust_events):
        # NDDI spike on the 8-day composite covering this dust day
        j = mod09_dates.searchsorted(dates[i], side="right") - 1
        if 0 <= j < len(mod09_df):
            mod09_df.iloc[j, nddi_col] = rng.uniform(0.05, 0.30)
        # Low-visibility window across the dust day
        day0 = dates[i]
        mask = (vis_df.index >= day0) & (
            vis_df.index <= day0 + pd.Timedelta(hours=23)
        )
        vis_df.loc[mask, "visibility_m"] = rng.uniform(200, 900, int(mask.sum()))

    # --- Static soil (typical Saudi desert values) ---
    sand_frac = rng.uniform(750, 900)
    clay = rng.uniform(30, 80)
    soil_feats = {
        "soil_clay_0-5cm": clay,
        "soil_sand_0-5cm": sand_frac,
        "soil_silt_0-5cm": 1000 - sand_frac - clay,
        "soil_ocs_0-5cm": rng.uniform(2, 8),
        "soil_bdod_0-5cm": rng.uniform(130, 160),
    }

    vis_flag = daily_visibility_flag(vis_df, threshold_m=1000.0)

    return {
        "albedo": albedo_df,
        "era5": era5,
        "mod09": mod09_df,
        "vis_flag": vis_flag,
        "soil_feats": soil_feats,
    }


def generate_all_synthetic_data(
    config: dict,
    output_dir: str | Path,
) -> None:
    """Write synthetic CSVs for all stations to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stations = config["stations"]
    syn = config.get("synthetic", {})
    seed = syn.get("seed", 42)
    positive_rate = syn.get("positive_rate", 0.10)
    start = f"{min(config['baseline_years'])}-01-01"
    end = f"{max(config['study_years'])}-12-31"

    soil_rows = []
    for station_idx, (name, coords) in enumerate(stations.items()):
        data = generate_synthetic_station_data(
            name,
            coords["lat"],
            coords["lon"],
            start=start,
            end=end,
            seed=seed,
            positive_rate=positive_rate,
            station_idx=station_idx,
        )
        data["albedo"].to_csv(output_dir / f"albedo_{name}.csv")
        data["era5"].to_csv(output_dir / f"era5_{name}_daily.csv")
        data["mod09"].to_csv(output_dir / f"mod09a1_{name}.csv")
        data["vis_flag"].to_csv(output_dir / f"vis_flag_{name}.csv")
        soil_row = {**data["soil_feats"], "station": name}
        soil_rows.append(soil_row)
        print(f"  Generated synthetic data for {name}")

    pd.DataFrame(soil_rows).set_index("station").to_csv(
        output_dir / "soil_properties.csv"
    )


def load_synthetic_station_bundle(
    station_name: str,
    data_dir: str | Path,
) -> dict[str, Any]:
    """Load pre-generated synthetic CSVs for one station."""
    data_dir = Path(data_dir)
    albedo = pd.read_csv(
        data_dir / f"albedo_{station_name}.csv", index_col="date", parse_dates=True
    )
    era5 = pd.read_csv(
        data_dir / f"era5_{station_name}_daily.csv", index_col="date", parse_dates=True
    )
    mod09 = pd.read_csv(
        data_dir / f"mod09a1_{station_name}.csv", index_col="date", parse_dates=True
    )
    vis_flag = pd.read_csv(
        data_dir / f"vis_flag_{station_name}.csv",
        index_col=0,
        parse_dates=True,
    ).squeeze()
    if isinstance(vis_flag, pd.DataFrame):
        vis_flag = vis_flag.iloc[:, 0]
    vis_flag.name = "vis_dust_flag"

    soil_df = pd.read_csv(data_dir / "soil_properties.csv", index_col="station")
    soil_feats = soil_df.loc[station_name].to_dict()

    return {
        "albedo": albedo,
        "era5": era5,
        "mod09": mod09,
        "vis_flag": vis_flag,
        "soil_feats": soil_feats,
    }
