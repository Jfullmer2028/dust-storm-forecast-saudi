"""
Model evaluation, statistical comparison, and visualisation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import fbeta_score

from src.models import TARGET


def per_station_f2(results: dict) -> pd.DataFrame:
    """PR-AUC, F2, precision and recall per station over out-of-fold predictions."""
    from sklearn.metrics import average_precision_score, precision_score, recall_score

    if "fold_station" not in results:
        return pd.DataFrame()

    y_true = np.concatenate(results["fold_true"])
    y_pred = np.concatenate(results["fold_preds"])
    proba = np.concatenate(results["fold_proba"])
    stations = np.concatenate(results["fold_station"])

    rows = []
    for st in sorted(np.unique(stations)):
        m = stations == st
        has_both = 0 < y_true[m].sum() < m.sum()
        rows.append(
            {
                "station": st,
                "n": int(m.sum()),
                "positives": int(y_true[m].sum()),
                "ap": average_precision_score(y_true[m], proba[m])
                if has_both
                else float("nan"),
                "f2": fbeta_score(y_true[m], y_pred[m], beta=2, zero_division=0),
                "precision": precision_score(y_true[m], y_pred[m], zero_division=0),
                "recall": recall_score(y_true[m], y_pred[m], zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_auc_ci(
    y_true: np.ndarray,
    proba_baseline: np.ndarray,
    proba_full: np.ndarray,
    metric: str = "ap",
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Threshold-free bootstrap CI for the change in a ranking metric.

    metric='ap'  -> PR-AUC (average precision); 'roc' -> ROC-AUC.

    Operates on the out-of-fold predicted probabilities, so it is unaffected by
    the per-fold decision threshold — the cleaner comparison for a rare-event
    forecaster. Paired resampling (same indices for both prediction sets)
    isolates the difference between them.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    score = average_precision_score if metric == "ap" else roc_auc_score
    rng = np.random.default_rng(seed=random_state)
    n = len(y_true)

    base_point = float(score(y_true, proba_baseline))
    full_point = float(score(y_true, proba_full))

    deltas = np.empty(n_bootstrap)
    filled = 0
    attempts = 0
    while filled < n_bootstrap and attempts < n_bootstrap * 20:
        attempts += 1
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue  # metric undefined without both classes
        deltas[filled] = score(yb, proba_full[idx]) - score(yb, proba_baseline[idx])
        filled += 1
    deltas = deltas[:filled]

    alpha = 1 - confidence_level
    lo = float(np.percentile(deltas, 100 * alpha / 2))
    hi = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    name = "PR-AUC" if metric == "ap" else "ROC-AUC"
    print(
        f"{name}: baseline={base_point:.4f}  full={full_point:.4f}  "
        f"delta={full_point - base_point:+.4f}  "
        f"{int(confidence_level * 100)}% CI [{lo:+.4f}, {hi:+.4f}]"
    )
    return {
        "metric": name,
        "baseline": base_point,
        "full": full_point,
        "delta": full_point - base_point,
        "lo": lo,
        "hi": hi,
        "samples": deltas,
    }


def benjamini_hochberg(pvals: np.ndarray, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg FDR control. Returns (reject, adjusted_pvalues)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    # BH-adjusted p-values (monotone from the largest rank down)
    adj = ranked * m / (np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out_adj = np.empty(m)
    out_adj[order] = adj
    reject = out_adj <= q
    return reject, out_adj


def run_group_ablation(
    df: pd.DataFrame,
    full_features: list[str],
    n_splits: int = 5,
    xgb_params: dict | None = None,
    random_state: int = 42,
    n_bootstrap: int = 2000,
    output_dir: str | Path = "outputs",
    full_results: dict | None = None,
    fdr_q: float = 0.05,
) -> pd.DataFrame:
    """
    Quantify the incremental value of each physical driver group.

    For each group g, retrain on (all features − g) and measure the drop in
    PR-AUC relative to the full model on the same out-of-fold predictions:

        incremental(g) = PR-AUC(all) − PR-AUC(all − g)

    Each group gets a paired bootstrap 95% CI and a two-sided bootstrap p-value.
    Because one group test is run per driver, p-values are corrected for multiple
    comparisons with **Benjamini-Hochberg FDR**; `significant_fdr` (not the raw
    CI) is the reported call.
    """
    from src.features import build_feature_groups
    from src.models import run_cross_validation

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = build_feature_groups(full_features)

    print("\n" + "=" * 60)
    print("DRIVER ABLATION — incremental PR-AUC by feature group (BH-FDR)")
    print("=" * 60)

    full_res = full_results or run_cross_validation(
        df, full_features, n_splits=n_splits, random_state=random_state,
        xgb_params=xgb_params, verbose=False,
    )
    y_true = np.concatenate(full_res["fold_true"])
    proba_full = np.concatenate(full_res["fold_proba"])

    rows = []
    for gname, gfeats in groups.items():
        reduced = [f for f in full_features if f not in set(gfeats)]
        if not reduced:
            continue
        res = run_cross_validation(
            df, reduced, n_splits=n_splits, random_state=random_state,
            xgb_params=xgb_params, verbose=False,
        )
        proba_reduced = np.concatenate(res["fold_proba"])
        # delta = full − reduced = the group's incremental PR-AUC contribution.
        cmp = bootstrap_auc_ci(
            y_true, proba_reduced, proba_full, metric="ap",
            n_bootstrap=n_bootstrap, random_state=random_state,
        )
        samples = cmp["samples"]
        # Two-sided bootstrap p-value for H0: delta = 0.
        p = 2 * min((samples <= 0).mean(), (samples >= 0).mean())
        p = float(np.clip(p, 1.0 / len(samples), 1.0))
        rows.append(
            {
                "group": gname,
                "n_features": len(gfeats),
                "incremental_pr_auc": cmp["delta"],
                "ci_lo": cmp["lo"],
                "ci_hi": cmp["hi"],
                "p_value": p,
            }
        )

    table = pd.DataFrame(rows).sort_values(
        "incremental_pr_auc", ascending=False
    ).reset_index(drop=True)

    reject, p_adj = benjamini_hochberg(table["p_value"].values, q=fdr_q)
    table["p_fdr"] = p_adj
    table["significant_fdr"] = reject

    for _, r in table.iterrows():
        print(
            f"  {r['group']:<20} ΔPR-AUC={r['incremental_pr_auc']:+.4f} "
            f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  "
            f"p={r['p_value']:.3f} p_fdr={r['p_fdr']:.3f}"
            f"{'  *' if r['significant_fdr'] else ''}"
        )

    # Horizontal bar chart with 95% CI whiskers (green = FDR-significant).
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(table) + 1.5))
    y = np.arange(len(table))[::-1]
    colors = ["#2E7D32" if s else "#9E9E9E" for s in table["significant_fdr"]]
    err_lo = (table["incremental_pr_auc"] - table["ci_lo"]).clip(lower=0)
    err_hi = (table["ci_hi"] - table["incremental_pr_auc"]).clip(lower=0)
    ax.barh(y, table["incremental_pr_auc"], color=colors,
            xerr=[err_lo, err_hi], capsize=3, alpha=0.9)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(table["group"])
    ax.set_xlabel("Incremental PR-AUC (full − without group)")
    ax.set_title("Driver ablation for 24-h dust onset\n"
                 f"(green = significant at BH-FDR q={fdr_q})")
    plt.tight_layout()
    plt.savefig(output_dir / "driver_ablation.png", dpi=150)
    plt.close()

    table.to_csv(output_dir / "driver_ablation.csv", index=False)
    return table


def compute_naive_baselines(
    df: pd.DataFrame,
    full_features: list[str],
    model_results: dict,
    n_splits: int = 5,
    xgb_params: dict | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Contextualise the forecaster against naive baselines (EMBRACE-style).

    Rows: no-skill (climatological base rate), persistence (dust tomorrow if dust
    today), meteorology-only model, and the full model. PR-AUC / ROC-AUC.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    from src.features import assign_group
    from src.models import TARGET, run_cross_validation

    d = df.sort_values(["station", "date"]).reset_index(drop=True)
    y = d[TARGET].astype(int).values
    base_rate = float(y.mean())

    # Persistence: predict next-day event from today's event (per station).
    persist = d.groupby("station")[TARGET].shift(1).fillna(0).astype(float).values
    persist_ap = average_precision_score(y, persist)
    persist_roc = roc_auc_score(y, persist)

    # Meteorology-only model (drop satellite vegetation/albedo and static soil).
    met_groups = {"wind_speed", "wind_direction", "humidity_dryness",
                  "antecedent_moisture", "thermal_blh", "pressure", "seasonality"}
    met_feats = [f for f in full_features if assign_group(f) in met_groups]
    met = run_cross_validation(
        d, met_feats, n_splits=n_splits, random_state=random_state,
        xgb_params=xgb_params, verbose=False,
    )

    rows = [
        {"baseline": "no-skill (base rate)", "pr_auc": base_rate, "roc_auc": 0.5},
        {"baseline": "persistence", "pr_auc": persist_ap, "roc_auc": persist_roc},
        {"baseline": "meteorology-only model", "pr_auc": met["mean_ap"],
         "roc_auc": met["mean_roc"]},
        {"baseline": "full model", "pr_auc": model_results["mean_ap"],
         "roc_auc": model_results["mean_roc"]},
    ]
    return pd.DataFrame(rows)


def operational_metrics(model_results: dict) -> dict[str, Any]:
    """
    Decision-relevant metrics on the out-of-fold probabilities, so a modest
    forecaster can still be judged as a *warning system*:

      - Brier score and Brier Skill Score (BSS) vs a climatology forecast
      - recall achievable at a usable precision (>= 0.30)
      - precision and false-alarm rate at the operating point that catches
        half of all dust days (recall = 0.50)
    """
    from sklearn.metrics import (
        brier_score_loss,
        precision_recall_curve,
        roc_curve,
    )

    y = np.concatenate(model_results["fold_true"])
    p = np.concatenate(model_results["fold_proba"])
    base = float(y.mean())

    brier = float(brier_score_loss(y, p))
    brier_clim = base * (1 - base)
    bss = 1 - brier / brier_clim if brier_clim > 0 else float("nan")

    prec, rec, _ = precision_recall_curve(y, p)
    rec_at_p30 = float(rec[prec >= 0.30].max()) if (prec >= 0.30).any() else 0.0
    prec_at_r50 = float(prec[rec >= 0.50].max()) if (rec >= 0.50).any() else 0.0

    fpr, tpr, _ = roc_curve(y, p)
    fpr_at_r50 = float(fpr[tpr >= 0.50][0]) if (tpr >= 0.50).any() else 1.0

    return {
        "base_rate": base,
        "brier": brier,
        "brier_skill_score": float(bss),
        "recall_at_precision30": rec_at_p30,
        "precision_at_recall50": prec_at_r50,
        "fpr_at_recall50": fpr_at_r50,
    }


def seed_robustness(
    df: pd.DataFrame,
    full_features: list[str],
    top_group_features: list[str],
    seeds: list[int],
    n_splits: int = 5,
    xgb_params: dict | None = None,
) -> dict[str, Any]:
    """
    Repeat the model and the top-driver ablation over several seeds and report
    mean ± sd, so the headline does not hinge on a single random seed.
    """
    from src.models import run_cross_validation

    ap, roc, top_delta = [], [], []
    reduced = [f for f in full_features if f not in set(top_group_features)]
    for s in seeds:
        full_s = run_cross_validation(
            df, full_features, n_splits=n_splits, random_state=s,
            xgb_params=xgb_params, verbose=False,
        )
        red_s = run_cross_validation(
            df, reduced, n_splits=n_splits, random_state=s,
            xgb_params=xgb_params, verbose=False,
        )
        ap.append(full_s["mean_ap"])
        roc.append(full_s["mean_roc"])
        top_delta.append(full_s["mean_ap"] - red_s["mean_ap"])
    return {
        "seeds": seeds,
        "ap_mean": float(np.mean(ap)), "ap_sd": float(np.std(ap)),
        "roc_mean": float(np.mean(roc)), "roc_sd": float(np.std(roc)),
        "top_delta_mean": float(np.mean(top_delta)),
        "top_delta_sd": float(np.std(top_delta)),
    }


def plot_calibration(results: dict, output_dir: str | Path = "outputs") -> None:
    """Reliability diagram for the forecast model's out-of-fold probabilities."""
    from sklearn.calibration import calibration_curve

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.concatenate(results["fold_true"])
    proba = np.concatenate(results["fold_proba"])
    n_pos = int(y_true.sum())
    if n_pos < 15:  # too few events for a meaningful reliability diagram
        print(f"  [calibration] only {n_pos} positives — skipping reliability plot")
        return
    n_bins = int(np.clip(n_pos // 15, 3, 10))
    frac_pos, mean_pred = calibration_curve(
        y_true, proba, n_bins=n_bins, strategy="quantile"
    )
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color="#E07B39", label="forecast model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed dust frequency")
    ax.set_title("Reliability diagram (out-of-fold)")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "calibration.png", dpi=150)
    plt.close()


def plot_pr_curve(results: dict, output_dir: str | Path = "outputs") -> None:
    """Precision-recall curve for the forecast model (out-of-fold predictions)."""
    from sklearn.metrics import average_precision_score, precision_recall_curve

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.concatenate(results["fold_true"])
    proba = np.concatenate(results["fold_proba"])
    base_rate = y_true.mean()
    ap = average_precision_score(y_true, proba)

    prec, rec, _ = precision_recall_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color="#E07B39", lw=2, label=f"Forecast model (AP={ap:.3f})")
    ax.axhline(base_rate, color="grey", ls="--", lw=1,
               label=f"No-skill ({base_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall — 24-hour dust-onset forecast")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=150)
    plt.close()


def plot_feature_importance(
    df: pd.DataFrame,
    feature_cols: list[str],
    output_dir: str | Path = "outputs",
    top_n: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train on all data and plot SHAP feature importances."""
    import shap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_sorted = df.sort_values(["station", "date"]).reset_index(drop=True)
    X = df_sorted[feature_cols].values.astype(float)
    y = df_sorted[TARGET].values.astype(int)

    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    col_med = np.nanmedian(X, axis=0)
    X_imp = np.where(np.isnan(X), np.tile(col_med, (X.shape[0], 1)), X)

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_imp, y, verbose=False)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_imp)

    mean_shap = np.abs(shap_vals).mean(axis=0)
    importance_df = pd.DataFrame(
        {"feature": feature_cols, "mean_shap": mean_shap}
    ).sort_values("mean_shap", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        importance_df["feature"][::-1],
        importance_df["mean_shap"][::-1],
        color="#5B8DB8",
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {top_n} Features by SHAP Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_importance.png", dpi=150)
    plt.close()

    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    return importance_df


def write_report(
    model_results: dict,
    ablation_df: pd.DataFrame,
    df: pd.DataFrame,
    output_path: str | Path = "results/report.md",
    data_mode: str = "synthetic",
    baselines_df: "pd.DataFrame | None" = None,
    seed_robust: dict | None = None,
    operational: dict | None = None,
) -> None:
    """Write the markdown results report for the driver study."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = len(df)
    n_positive = int(df[TARGET].sum())
    pos_rate = 100 * n_positive / n_samples if n_samples else 0
    n_folds = len(model_results["fold_f2"])

    seed_line = ""
    if seed_robust is not None:
        seed_line = (
            f" Across {len(seed_robust['seeds'])} random seeds, PR-AUC = "
            f"{seed_robust['ap_mean']:.3f} ± {seed_robust['ap_sd']:.3f} and "
            f"ROC-AUC = {seed_robust['roc_mean']:.3f} ± {seed_robust['roc_sd']:.3f}."
        )

    lines = [
        "# Dust-Storm Onset Forecasting — Results Report",
        "",
        f"**Data mode:** {data_mode}",
        "",
        "## Dataset Summary",
        "",
        f"- Total samples: {n_samples:,}",
        f"- Positive (dust event next day): {n_positive:,} ({pos_rate:.1f}%)",
        f"- Stations: {', '.join(sorted(df['station'].unique()))}",
        f"- Study period: {df['date'].min()} to {df['date'].max()}",
        "",
        f"## Forecast Model Performance (TimeSeriesSplit, {n_folds} folds)",
        "",
        "Out-of-fold cross-validated skill of the XGBoost forecaster. PR-AUC "
        "(average precision) is the primary metric for this rare-event problem; "
        "ROC-AUC and F2 (at a per-fold tuned threshold) are reported alongside."
        + seed_line,
        "",
        "| Metric | Mean | Std |",
        "|--------|------|-----|",
        f"| PR-AUC | {model_results['mean_ap']:.4f} | {model_results['std_ap']:.4f} |",
        f"| ROC-AUC | {model_results['mean_roc']:.4f} | — |",
        f"| F2 (operational) | {model_results['mean_f2']:.4f} | {model_results['std_f2']:.4f} |",
        "",
        "| Fold | PR-AUC | ROC-AUC | F2 |",
        "|------|--------|---------|----|",
    ]
    for i in range(n_folds):
        lines.append(
            f"| {i + 1} | {model_results['fold_ap'][i]:.4f} | "
            f"{model_results['fold_roc'][i]:.4f} | "
            f"{model_results['fold_f2'][i]:.4f} |"
        )
    lines.append("")

    # Naive baselines (EMBRACE-style context)
    if baselines_df is not None and not baselines_df.empty:
        lines.extend(
            [
                "## Naive Baselines",
                "",
                "The forecaster against simple references (out-of-fold where a "
                "model is involved):",
                "",
                "| Reference | PR-AUC | ROC-AUC |",
                "|-----------|--------|---------|",
            ]
        )
        for _, r in baselines_df.iterrows():
            lines.append(
                f"| {r['baseline']} | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} |"
            )
        lines.append("")

    # Operational / decision-relevant evaluation
    if operational is not None:
        o = operational
        lines.extend(
            [
                "## Operational Evaluation",
                "",
                "Judging the probabilities as a warning system (out-of-fold):",
                "",
                "| Quantity | Value |",
                "|----------|-------|",
                f"| Brier score | {o['brier']:.4f} |",
                f"| Brier Skill Score (vs climatology) | {o['brier_skill_score']:+.3f} |",
                f"| Recall at precision ≥ 0.30 | {o['recall_at_precision30']:.2f} |",
                f"| Precision at recall = 0.50 | {o['precision_at_recall50']:.2f} |",
                f"| False-alarm rate at recall = 0.50 | {o['fpr_at_recall50']:.2f} |",
                "",
                f"To catch **half of all dust days**, the model issues warnings "
                f"on {100 * o['fpr_at_recall50']:.0f}% of calm days "
                f"(precision {o['precision_at_recall50']:.2f} at a "
                f"{100 * o['base_rate']:.1f}% base rate — a "
                f"{o['precision_at_recall50'] / max(o['base_rate'], 1e-9):.1f}× "
                "lift over random). "
                + (
                    "The calibrated probabilities are sharper than a climatology "
                    "forecast (positive Brier Skill Score)."
                    if o["brier_skill_score"] > 0
                    else "The Brier Skill Score is not above climatology, so the "
                    "probabilities add ranking value but limited probabilistic "
                    "sharpness over the base rate."
                ),
                "",
            ]
        )

    # Driver ablation — the headline analysis (BH-FDR corrected)
    sig = ablation_df[ablation_df["significant_fdr"]]
    lines.extend(
        [
            "## Driver Ablation (BH-FDR corrected)",
            "",
            "Incremental skill of each physical driver group: the change in "
            "PR-AUC when that group is removed and the model retrained "
            "(model − without-group), with paired bootstrap 95% CIs and "
            "two-sided bootstrap p-values. Because one test is run per driver "
            "group, p-values are corrected for multiple comparisons with "
            "**Benjamini-Hochberg FDR**; the corrected call (`sig.`) is the "
            "reported result.",
            "",
            "| Driver group | # feats | Incremental PR-AUC | 95% CI | p | p (FDR) | sig. |",
            "|--------------|---------|--------------------|--------|---|---------|------|",
        ]
    )
    for _, r in ablation_df.iterrows():
        lines.append(
            f"| {r['group']} | {int(r['n_features'])} | "
            f"{r['incremental_pr_auc']:+.4f} | "
            f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['p_value']:.3f} | "
            f"{r['p_fdr']:.3f} | {'**yes**' if r['significant_fdr'] else 'no'} |"
        )
    top_row = ablation_df.iloc[0]
    if not sig.empty:
        drivers = ", ".join(sig["group"].tolist())
        lines += [
            "",
            f"**Driver groups significant after FDR correction:** {drivers}.",
        ]
    else:
        lines += [
            "",
            f"**No driver group remains significant after FDR correction.** "
            f"The largest incremental contribution is **{top_row['group']}** "
            f"(ΔPR-AUC {top_row['incremental_pr_auc']:+.4f}, raw p="
            f"{top_row['p_value']:.3f}, FDR p={top_row['p_fdr']:.3f}) — "
            "suggestive but not confirmed at this sample size.",
        ]
    if seed_robust is not None:
        lines += [
            "",
            f"Seed robustness of the top driver "
            f"(ΔPR-AUC over {len(seed_robust['seeds'])} seeds): "
            f"{seed_robust['top_delta_mean']:+.4f} ± {seed_robust['top_delta_sd']:.4f}.",
        ]
    lines.append("")

    # Per-station performance
    ps = per_station_f2(model_results)
    if not ps.empty:
        lines.extend(
            [
                "## Per-Station Performance",
                "",
                "| Station | n | Positives | PR-AUC | F2 | Precision | Recall |",
                "|---------|---|-----------|--------|----|-----------|--------|",
            ]
        )
        for _, r in ps.iterrows():
            lines.append(
                f"| {r['station']} | {int(r['n'])} | {int(r['positives'])} | "
                f"{r['ap']:.4f} | {r['f2']:.4f} | {r['precision']:.3f} | "
                f"{r['recall']:.3f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Figures",
            "",
            "- `driver_ablation.png` — incremental PR-AUC by driver group (FDR)",
            "- `pr_curve.png` — precision–recall curve for the forecast model",
            "- `calibration.png` — reliability diagram (out-of-fold)",
            "- `shap_importance.png` — SHAP feature importance",
            "",
            "## Conclusion",
            "",
        ]
    )

    intro = (
        f"The forecaster attains a cross-validated PR-AUC of "
        f"{model_results['mean_ap']:.3f} (ROC-AUC {model_results['mean_roc']:.3f}) "
        f"at a {pos_rate:.1f}% base rate, well above the no-skill PR-AUC of "
        f"{pos_rate / 100:.3f}. "
    )
    if not sig.empty:
        drivers = ", ".join(sig["group"].tolist())
        top = sig.iloc[0]
        lines.append(
            intro + f"After Benjamini-Hochberg FDR correction across the driver "
            f"groups, **{drivers}** retain{'s' if len(sig) == 1 else ''} "
            f"statistically significant incremental skill, with **{top['group']}** "
            f"the strongest (ΔPR-AUC {top['incremental_pr_auc']:+.4f}, FDR "
            f"p={top['p_fdr']:.3f}). Remaining groups carry information already "
            "present elsewhere in the feature set."
        )
    else:
        lines.append(
            intro + f"After Benjamini-Hochberg FDR correction across the driver "
            f"groups, no single group reaches significance at this sample size; "
            f"the strongest incremental contribution is **{top_row['group']}** "
            f"(ΔPR-AUC {top_row['incremental_pr_auc']:+.4f}, FDR p="
            f"{top_row['p_fdr']:.3f}), which we report as suggestive. Widening the "
            "study (more stations/years via `--stations` / `--modis-years`) is the "
            "natural way to confirm it."
        )

    output_path.write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {output_path}")
