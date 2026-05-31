"""
Dataset construction and dust-event labeling.
"""

from __future__ import annotations

import pandas as pd

from src.features import (
    ERA5_LAG_COLS,
    add_lag_features,
    add_temporal_features,
)


def build_nddi_daily(mod09_df: pd.DataFrame) -> pd.Series:
    """
    Expand 8-day MOD09A1 NDDI composite to daily resolution via forward-fill
    (limit 7 days per composite period).
    """
    daily_idx = pd.date_range(
        mod09_df.index.min(),
        mod09_df.index.max(),
        freq="D",
    )
    nddi_daily = mod09_df["nddi"].reindex(daily_idx).ffill(limit=7)
    return nddi_daily


def build_labels(
    nddi_daily: pd.Series,
    vis_flag: pd.Series,
) -> pd.Series:
    """
    Binary dust-event label:
      1 if NDDI > 0 AND visibility <= 1 km on the same day
      0 otherwise
    """
    aligned = pd.DataFrame(
        {
            "nddi_pos": nddi_daily > 0,
            "vis_dust": vis_flag,
        }
    ).dropna(subset=["nddi_pos"])

    aligned["vis_dust"] = aligned["vis_dust"].fillna(False)
    label = (aligned["nddi_pos"] & aligned["vis_dust"]).astype(int)
    label.name = "dust_event"
    return label


def build_master_dataframe(
    station_name: str,
    era5_df: pd.DataFrame,
    albedo_df: pd.DataFrame,
    mod09_df: pd.DataFrame,
    vis_flag: pd.Series,
    soil_feats: dict,
    study_start: str = "2018",
    study_end: str = "2020",
) -> pd.DataFrame:
    """
    Merge all feature sources and the next-day dust label.

    Features on day D predict dust onset on day D+1 (label shifted -1).
    """
    nddi_daily = build_nddi_daily(mod09_df)
    label = build_labels(nddi_daily, vis_flag)
    label_shifted = label.shift(-1)
    label_shifted.name = "dust_event_next_day"

    df = era5_df.copy()
    df.index = pd.to_datetime(df.index)

    alb_cols = [
        "albedo_wsb",
        "albedo_anomaly",
        "albedo_anom_3d",
        "albedo_anom_7d",
    ]
    available_alb = [c for c in alb_cols if c in albedo_df.columns]
    if available_alb:
        df = df.join(albedo_df[available_alb], how="left")

    ndvi_daily = mod09_df["ndvi"].reindex(df.index).ffill(limit=7)
    df["ndvi"] = ndvi_daily

    df = add_temporal_features(df)
    df = add_lag_features(df, ERA5_LAG_COLS)

    for feat_name, feat_val in soil_feats.items():
        if feat_name != "station":
            df[feat_name] = feat_val

    df["station"] = station_name
    df = df.join(label_shifted, how="left")
    df = df.loc[study_start:study_end].dropna(subset=["dust_event_next_day"])

    return df


def build_full_dataset(all_station_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine station DataFrames; forward-fill albedo gaps <= 3 days."""
    combined = pd.concat(all_station_dfs, axis=0)
    combined = combined.reset_index().rename(columns={"index": "date"})
    if "date" not in combined.columns and "index" in combined.columns:
        combined = combined.rename(columns={"index": "date"})

    albedo_feature_cols = [
        "albedo_wsb",
        "albedo_anomaly",
        "albedo_anom_3d",
        "albedo_anom_7d",
    ]
    present = [c for c in albedo_feature_cols if c in combined.columns]
    if present:
        combined[present] = (
            combined.sort_values(["station", "date"])
            .groupby("station")[present]
            .transform(lambda x: x.ffill(limit=3))
        )

    return combined.sort_values(["station", "date"]).reset_index(drop=True)
