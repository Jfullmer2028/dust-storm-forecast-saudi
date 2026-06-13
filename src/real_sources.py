"""
Keyless real-data acquisition.

Every source here is a public API that needs **no account, key, or OAuth**, so
the real-data pipeline runs in any environment with outbound network access:

* **Meteorology** — Open-Meteo Historical Weather API, which serves ERA5
  reanalysis for any point (https://open-meteo.com/en/docs/historical-weather-api).
* **Satellite** — ORNL DAAC MODIS/VIIRS Land Product Subsets REST API
  (https://modis.ornl.gov/data/modis_webservice.html). We pull MOD09A1 8-day
  surface reflectance and derive NDVI, the NDDI dust index, and a shortwave
  broadband **albedo** via the Liang (2001) narrow-to-broadband conversion.
* **Visibility** — NOAA Integrated Surface Database "global-hourly" CSV access
  (https://www.ncei.noaa.gov/data/global-hourly/access/).
* **Soil** — ISRIC SoilGrids v2 REST API (reused from ``acquisition``).

This is a credential-free alternative to the Google Earth Engine + Copernicus
CDS path in ``acquisition.py``; that path remains available for users who
prefer MCD43A3 albedo and the native ERA5 archive.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.acquisition import daily_visibility_flag, fetch_soilgrids

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
ORNL_URL = "https://modis.ornl.gov/rst/api/v1"
ISD_CSV_URL = "https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{station}.csv"

# Liang (2001) MODIS surface-reflectance -> shortwave broadband albedo (0.3-3.0 um)
LIANG_COEFFS = {
    "sur_refl_b01": 0.160,
    "sur_refl_b02": 0.291,
    "sur_refl_b03": 0.243,
    "sur_refl_b04": 0.116,
    "sur_refl_b05": 0.112,
    "sur_refl_b07": 0.081,
}
LIANG_INTERCEPT = -0.0015
MOD09A1_BANDS = [f"sur_refl_b0{i}" for i in range(1, 8)]  # b01..b07


def _get(
    url: str,
    params: dict | None = None,
    timeout: int = 90,
    retries: int = 5,
    headers: dict | None = None,
) -> requests.Response:
    """GET with exponential backoff on transient failures / rate limits."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=headers)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(2 ** attempt * 1.5, 30))
                continue
            r.raise_for_status()
        except requests.RequestException as exc:  # network hiccup
            last_exc = exc
            time.sleep(min(2 ** attempt * 1.5, 30))
    raise RuntimeError(f"GET failed after {retries} attempts: {url}") from last_exc


# ---------------------------------------------------------------------------
# Open-Meteo ERA5 meteorology (no key)
# ---------------------------------------------------------------------------

OPEN_METEO_HOURLY = [
    "temperature_2m",
    "dew_point_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_gusts_10m",
    "precipitation",
    "soil_temperature_0_to_7cm",
    "soil_moisture_0_to_7cm",
    "cloud_cover",
]


def fetch_openmeteo_daily(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Fetch ERA5 hourly fields from Open-Meteo and reduce to daily features."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(OPEN_METEO_HOURLY),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }
    data = _get(OPEN_METEO_URL, params).json()["hourly"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index().astype(
        {c: "float64" for c in OPEN_METEO_HOURLY}
    )

    agg = df.resample("D").agg(
        ws_max=("wind_speed_10m", "max"),
        ws_mean=("wind_speed_10m", "mean"),
        gust_max=("wind_gusts_10m", "max"),
        rh_mean=("relative_humidity_2m", "mean"),
        rh_min=("relative_humidity_2m", "min"),
        t2m_mean=("temperature_2m", "mean"),
        t2m_max=("temperature_2m", "max"),
        td_mean=("dew_point_2m", "mean"),
        sp_mean=("surface_pressure", "mean"),
        sm_mean=("soil_moisture_0_to_7cm", "mean"),
        soilt_mean=("soil_temperature_0_to_7cm", "mean"),
        cloud_mean=("cloud_cover", "mean"),
        precip_sum=("precipitation", "sum"),
    )
    agg["precip_7d"] = agg["precip_sum"].rolling(7, min_periods=1).sum().shift(1)

    # Vapour-pressure deficit (kPa) — a strong dryness/erodibility proxy
    es = 0.6108 * np.exp(17.27 * agg["t2m_mean"] / (agg["t2m_mean"] + 237.3))
    ea = 0.6108 * np.exp(17.27 * agg["td_mean"] / (agg["td_mean"] + 237.3))
    agg["vpd_mean"] = (es - ea).clip(lower=0)

    agg.index.name = "date"
    return agg


# ---------------------------------------------------------------------------
# NOAA ISD visibility (no key)
# ---------------------------------------------------------------------------

def fetch_isd_visibility(usaf: str, wban: str, year: int) -> pd.DataFrame:
    """Hourly visibility (metres) from the NOAA global-hourly CSV archive."""
    station = f"{usaf}{wban}"
    url = ISD_CSV_URL.format(year=year, station=station)
    r = _get(url, timeout=180)
    raw = pd.read_csv(
        io.StringIO(r.text), usecols=["DATE", "VIS"], dtype=str, low_memory=False
    )
    dt = pd.to_datetime(raw["DATE"], errors="coerce")
    # VIS field: "distance,quality,variability,variability-quality"; first
    # sub-field is the horizontal visibility in metres (999999 == missing).
    vis_m = pd.to_numeric(raw["VIS"].str.split(",", expand=True)[0], errors="coerce")
    vis_m[vis_m >= 999999] = np.nan
    out = pd.DataFrame({"visibility_m": vis_m.values}, index=dt)
    out = out[out.index.notna()].sort_index()
    out.index.name = "datetime"
    return out


def fetch_visibility_flag(
    usaf: str, wban: str, years: list[int], threshold_m: float = 1000.0
) -> pd.Series:
    """Daily dust flag (any hourly visibility <= threshold) across years."""
    parts = []
    for y in years:
        try:
            parts.append(fetch_isd_visibility(usaf, wban, y))
        except Exception as exc:  # noqa: BLE001 - skip a missing station-year
            print(f"    [ISD] {usaf}-{wban} {y}: {exc}")
    if not parts:
        raise RuntimeError(f"No ISD visibility for {usaf}-{wban} in {years}")
    allv = pd.concat(parts).sort_index()
    return daily_visibility_flag(allv, threshold_m)


# ---------------------------------------------------------------------------
# ORNL DAAC MODIS MOD09A1 surface reflectance (no key)
# ---------------------------------------------------------------------------

def _ornl_dates(lat: float, lon: float, product: str = "MOD09A1") -> pd.DataFrame:
    data = _get(f"{ORNL_URL}/{product}/dates", {"latitude": lat, "longitude": lon}).json()
    df = pd.DataFrame(data["dates"])
    df["calendar_date"] = pd.to_datetime(df["calendar_date"])
    return df


def _ornl_band_mean(
    lat: float, lon: float, band: str, modis_date: str, km: int, product: str
) -> float:
    """Spatial-mean of one band over the +/-km box for one composite date."""
    j = _get(
        f"{ORNL_URL}/{product}/subset",
        {
            "latitude": lat,
            "longitude": lon,
            "band": band,
            "startDate": modis_date,
            "endDate": modis_date,
            "kmAboveBelow": km,
            "kmLeftRight": km,
        },
    ).json()
    if isinstance(j, str) or not j.get("subset"):
        return float("nan")
    scale = float(j.get("scale", 1.0) or 1.0)
    arr = np.asarray(j["subset"][0]["data"], dtype="float64") * scale
    # MOD09A1 reflectance valid range ~[-0.01, 1.6]; everything else is fill.
    arr[(arr < -0.05) | (arr > 1.6)] = np.nan
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def fetch_mod09a1_indices(
    lat: float,
    lon: float,
    start: str,
    end: str,
    km: int = 20,
    sleep_s: float = 0.15,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Build an 8-day MODIS frame with NDVI, NDDI and Liang shortwave albedo.

    One reflectance band requires one ORNL request, so each composite date
    costs seven requests; results are returned per available composite date.
    """
    dates = _ornl_dates(lat, lon)
    sel = dates[
        (dates["calendar_date"] >= pd.Timestamp(start))
        & (dates["calendar_date"] <= pd.Timestamp(end))
    ]
    rows = []
    for k, (_, d) in enumerate(sel.iterrows()):
        rec: dict[str, Any] = {"date": d["calendar_date"]}
        for band in MOD09A1_BANDS:
            rec[band] = _ornl_band_mean(
                lat, lon, band, d["modis_date"], km, "MOD09A1"
            )
            time.sleep(sleep_s)
        rows.append(rec)
        if progress and (k % 10 == 0 or k == len(sel) - 1):
            print(f"    [MODIS] {d['calendar_date'].date()} ({k + 1}/{len(sel)})")
    df = pd.DataFrame(rows).set_index("date").sort_index()

    red, nir, swir = df["sur_refl_b01"], df["sur_refl_b02"], df["sur_refl_b06"]
    df["ndvi"] = (nir - red) / (nir + red)
    ndwi = (nir - swir) / (nir + swir)
    df["nddi"] = (df["ndvi"] - ndwi) / (df["ndvi"] + ndwi)
    df["albedo_wsb"] = LIANG_INTERCEPT + sum(
        coeff * df[band] for band, coeff in LIANG_COEFFS.items()
    )
    return df


# ---------------------------------------------------------------------------
# Per-station assembly with on-disk caching
# ---------------------------------------------------------------------------

def fetch_station_real_data(
    station_name: str,
    coords: dict,
    isd_id: dict,
    study_years: list[int],
    modis_years: list[int],
    soil_properties: list[str],
    albedo_km: int = 20,
    cache_dir: str | Path = "data/raw/real",
) -> dict[str, Any]:
    """
    Fetch (or load from cache) every real source for one station.

    Returns the bundle expected by ``compute_albedo_anomaly`` /
    ``build_master_dataframe``: ``albedo`` (daily), ``era5`` (daily),
    ``mod09`` (8-day NDVI/NDDI), ``vis_flag`` (daily) and ``soil_feats``.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    lat, lon = coords["lat"], coords["lon"]

    # --- MODIS (the expensive part) — cache per station ---
    modis_path = cache / f"modis_{station_name}.csv"
    if modis_path.exists():
        modis = pd.read_csv(modis_path, index_col="date", parse_dates=True)
        print(f"  [{station_name}] MODIS from cache ({len(modis)} composites)")
    else:
        print(f"  [{station_name}] fetching MODIS {min(modis_years)}-{max(modis_years)}...")
        modis = fetch_mod09a1_indices(
            lat,
            lon,
            start=f"{min(modis_years)}-01-01",
            end=f"{max(modis_years)}-12-31",
            km=albedo_km,
        )
        modis.to_csv(modis_path)

    # Daily albedo (forward-fill 8-day composites across the period)
    daily_idx = pd.date_range(modis.index.min(), modis.index.max(), freq="D")
    albedo_daily = (
        modis[["albedo_wsb"]].reindex(daily_idx).ffill(limit=8)
    )
    albedo_daily.index.name = "date"
    albedo_daily["station"] = station_name

    # --- Meteorology — cache per station ---
    era5_path = cache / f"era5_{station_name}.csv"
    if era5_path.exists():
        era5 = pd.read_csv(era5_path, index_col="date", parse_dates=True)
        print(f"  [{station_name}] meteorology from cache ({len(era5)} days)")
    else:
        print(f"  [{station_name}] fetching Open-Meteo ERA5...")
        era5 = fetch_openmeteo_daily(
            lat, lon, f"{min(study_years)}-01-01", f"{max(study_years)}-12-31"
        )
        era5.to_csv(era5_path)

    # --- Visibility — cache per station ---
    vis_path = cache / f"vis_{station_name}.csv"
    if vis_path.exists():
        vis_flag = pd.read_csv(
            vis_path, index_col=0, parse_dates=True
        ).squeeze("columns")
        vis_flag.name = "vis_dust_flag"
        print(f"  [{station_name}] visibility from cache")
    else:
        print(f"  [{station_name}] fetching NOAA ISD visibility...")
        vis_flag = fetch_visibility_flag(
            isd_id["usaf"], isd_id["wban"], study_years
        )
        vis_flag.to_csv(vis_path)

    # --- Soil (cheap) — cache per station ---
    soil_path = cache / f"soil_{station_name}.csv"
    if soil_path.exists():
        soil_feats = pd.read_csv(soil_path).iloc[0].to_dict()
    else:
        print(f"  [{station_name}] fetching SoilGrids...")
        soil_feats = fetch_soilgrids(lat, lon, soil_properties)
        pd.DataFrame([soil_feats]).to_csv(soil_path, index=False)

    return {
        "albedo": albedo_daily,
        "era5": era5,
        "mod09": modis[["ndvi", "nddi"]].copy(),
        "vis_flag": vis_flag,
        "soil_feats": soil_feats,
    }
