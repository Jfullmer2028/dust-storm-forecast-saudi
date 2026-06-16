"""
Live interactive demo — Dust-Storm Forecast (Saudi Arabia).

Run locally:
    pip install -r requirements-demo.txt
    streamlit run app.py

Or deploy for free to Streamlit Community Cloud (https://streamlit.io/cloud):
point it at this repo and `app.py` — it is fully self-contained (rebuilds the
bundled synthetic dataset and trains the model on launch; no external data or
keys required).

Four tabs:
  1. Findings           — the headline result and driver-ablation chart
  2. Driver ablation    — incremental PR-AUC per driver group (interactive)
  3. What-if predictor  — move sliders, get a live next-day dust probability
  4. Station explorer    — predicted probability vs. observed dust events
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.acquisition import (
    generate_all_synthetic_data,
    load_config,
    load_synthetic_station_bundle,
)
from src.evaluation import run_group_ablation
from src.features import (
    FULL_FEATURES,
    compute_albedo_anomaly,
)
from src.labeling import build_full_dataset, build_master_dataframe
from src.models import TARGET, _make_xgb_classifier

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Dust-Storm Forecast — Saudi Arabia",
    page_icon="🌪️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Data + model (cached so the app is instant after first load)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Building dataset and training model…")
def load_everything():
    config = load_config(ROOT / "config.yaml")
    syn_dir = ROOT / config["paths"]["data_synthetic"]
    if not (syn_dir / "soil_properties.csv").exists():
        generate_all_synthetic_data(config, syn_dir)

    station_dfs = []
    for name in config["stations"]:
        bundle = load_synthetic_station_bundle(name, syn_dir)
        albedo_anom = compute_albedo_anomaly(
            bundle["albedo"],
            baseline_years=config["baseline_years"],
            study_years=config["study_years"],
            doy_window=config["project"]["albedo_doy_window"],
        )
        station_dfs.append(
            build_master_dataframe(
                name, bundle["era5"], albedo_anom, bundle["mod09"],
                bundle["vis_flag"], bundle["soil_feats"],
                study_start="2018", study_end="2022",
            )
        )
    df = build_full_dataset(station_dfs)

    feats = [c for c in FULL_FEATURES if c in df.columns]
    X = df[feats].astype(float)
    y = df[TARGET].astype(int).values
    medians = X.median()
    X_imp = X.fillna(medians).values
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    model = _make_xgb_classifier(spw, {"n_estimators": 300})
    model.fit(X_imp, y, verbose=False)
    return config, df, feats, medians, model


@st.cache_data(show_spinner="Running driver ablation…")
def get_ablation(_df, feats):
    # Prefer the precomputed table from a pipeline run; else compute now.
    csv = ROOT / "outputs" / "driver_ablation.csv"
    if csv.exists():
        return pd.read_csv(csv)
    return run_group_ablation(
        _df, feats, n_splits=5, xgb_params={"n_estimators": 150},
        n_bootstrap=800, output_dir=ROOT / "outputs",
    )


config, df, feats, medians, model = load_everything()

st.title("🌪️ Forecasting Dust-Storm Onset in Saudi Arabia")
st.caption(
    "24-hour-ahead dust onset · XGBoost · identifying the drivers that carry skill · "
    "interactive companion to the project (synthetic-data model for live use)"
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 Findings", "📊 Driver ablation", "🎛️ What-if predictor", "📈 Station explorer"]
)

# ---------------------------------------------------------------------------
# Tab 1 — Findings
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("What the study found")
    c1, c2, c3 = st.columns(3)
    c1.metric("Robust drivers (real)", "3 groups", "humidity · vegetation · seasonality")
    c2.metric("Forecast skill", "ROC-AUC 0.69", "PR-AUC 0.13 · BSS +0.012")
    c3.metric("Stations", "6", "6,570 station-days")
    st.markdown(
        """
A systematic, FDR-corrected *driver ablation* across **6 Saudi stations** ranks
each satellite and reanalysis driver group by its incremental forecasting skill.
Three groups survive multiple-comparison correction: **humidity/dryness,
vegetation cover (NDVI) and seasonality** — physically, dry air over a bare,
erodible surface during the dust season.

Honest caveat: absolute skill is modest and station-dependent, and the operating
points are weak (catching half of all dust days costs a ~38% false-alarm rate),
so this is a reproducible **baseline with robust driver attribution**, not a
deployable warning system.

This live demo trains on the project's **synthetic** generator (self-contained,
no keys) so you can interact with the model. The real-data numbers above come
from `--mode real` (Open-Meteo ERA5 + ORNL MODIS + NOAA ISD + SoilGrids); see
`results/report_real.md` and `PAPER.md`.
        """
    )
    img = ROOT / "outputs" / "real" / "driver_ablation.png"
    if img.exists():
        st.image(str(img), caption="Real-data driver ablation (green = 95% CI above zero)")

# ---------------------------------------------------------------------------
# Tab 2 — Driver ablation
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Incremental skill of each driver group")
    st.caption(
        "Drop one physical driver group, retrain, and measure the change in "
        "PR-AUC. P-values are Benjamini-Hochberg FDR-corrected across groups; "
        "`significant_fdr` is the corrected call. (Computed live on the "
        "synthetic model.)"
    )
    ab = get_ablation(df, feats)
    ab = ab.sort_values("incremental_pr_auc", ascending=False)
    st.bar_chart(ab.set_index("group")["incremental_pr_auc"], horizontal=True)
    cols = [c for c in ["group", "n_features", "incremental_pr_auc", "ci_lo",
                        "ci_hi", "p_value", "p_fdr", "significant_fdr"]
            if c in ab.columns]
    st.dataframe(
        ab[cols].style.format(
            {"incremental_pr_auc": "{:+.4f}", "ci_lo": "{:+.4f}",
             "ci_hi": "{:+.4f}", "p_value": "{:.3f}", "p_fdr": "{:.3f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------------------------
# Tab 3 — What-if predictor
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Live next-day dust-risk predictor")
    st.caption(
        "Move the dominant drivers; all other features stay at their median. "
        "Toggle albedo to see how little it moves the probability."
    )

    def slider_for(col, label, lo=None, hi=None):
        if col not in feats:
            return None
        series = df[col].astype(float)
        lo = float(series.quantile(0.02)) if lo is None else lo
        hi = float(series.quantile(0.98)) if hi is None else hi
        return st.slider(
            label, lo, hi, float(series.median()), (hi - lo) / 100 or 0.01
        )

    colA, colB = st.columns(2)
    overrides = {}
    with colA:
        overrides["ws_max"] = slider_for("ws_max", "Max wind speed (m/s)")
        overrides["northerly_frac"] = slider_for(
            "northerly_frac", "Northerly-flow fraction (shamal)", 0.0, 1.0
        )
        overrides["rh_mean"] = slider_for("rh_mean", "Mean relative humidity (%)")
        overrides["sm_mean"] = slider_for("sm_mean", "Soil moisture")
    with colB:
        overrides["ndvi"] = slider_for("ndvi", "NDVI (vegetation cover)")
        overrides["tcwv_mean"] = slider_for("tcwv_mean", "Column water vapour")
        overrides["precip_7d"] = slider_for("precip_7d", "7-day antecedent precip")

    x = medians.copy()
    for k, v in overrides.items():
        if v is not None and k in x.index:
            x[k] = v
    proba = float(model.predict_proba(x.values.reshape(1, -1))[:, 1][0])

    st.metric("Predicted probability of dust onset tomorrow", f"{proba * 100:.1f}%")
    st.progress(min(max(proba, 0.0), 1.0))
    base_rate = df[TARGET].mean()
    if proba > 3 * base_rate:
        st.error("Elevated dust risk — well above the climatological base rate.")
    elif proba > base_rate:
        st.warning("Moderately above-average dust risk.")
    else:
        st.success("Low dust risk (near or below base rate).")
    st.caption(f"Climatological base rate ≈ {base_rate * 100:.1f}%.")

# ---------------------------------------------------------------------------
# Tab 4 — Station explorer
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Predicted probability vs. observed dust events")
    station = st.selectbox("Station", sorted(df["station"].unique()))
    sdf = df[df["station"] == station].sort_values("date").copy()
    sdf["proba"] = model.predict_proba(
        sdf[feats].astype(float).fillna(medians).values
    )[:, 1]
    sdf["date"] = pd.to_datetime(sdf["date"])
    plot = sdf.set_index("date")[["proba", TARGET]].rename(
        columns={"proba": "Predicted P(dust)", TARGET: "Observed dust event"}
    )
    st.line_chart(plot["Predicted P(dust)"])
    events = sdf[sdf[TARGET] == 1]
    st.caption(
        f"{station}: {len(events)} observed next-day dust events out of "
        f"{len(sdf)} days ({100 * len(events) / len(sdf):.1f}%)."
    )
    st.scatter_chart(plot, y="Predicted P(dust)", x_label="date")
