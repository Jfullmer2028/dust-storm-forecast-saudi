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
from scipy.stats import wilcoxon
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


def wilcoxon_test(
    baseline_results: dict,
    full_results: dict,
) -> dict[str, Any]:
    """
    Wilcoxon signed-rank test on paired per-fold F2 scores.

    H0: median(F2_full - F2_baseline) = 0
    """
    d = full_results["fold_f2"] - baseline_results["fold_f2"]
    print(f"Per-fold F2 differences: {np.round(d, 4)}")

    if len(d) < 5:
        print("Warning: fewer than 5 folds — Wilcoxon test has very low power.")
        print("Interpret p-value cautiously; prefer bootstrap CI.")

    stat, p = wilcoxon(d, zero_method="zsplit", alternative="two-sided")
    print(f"Wilcoxon W = {stat:.2f},  p = {p:.4f}")
    if p < 0.05:
        print("-> Statistically significant at alpha=0.05")
    else:
        print("-> Not statistically significant at alpha=0.05")

    return {"W": stat, "p": p, "differences": d}


def bootstrap_f2_ci(
    true_labels: np.ndarray,
    baseline_preds: np.ndarray,
    full_preds: np.ndarray,
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95,
    random_state: int = 42,
    output_dir: str | Path = "outputs",
) -> dict[str, Any]:
    """
    Bootstrap confidence interval for Delta F2 = F2(full) - F2(baseline).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed=random_state)
    n = len(true_labels)
    delta_boot = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_b = true_labels[idx]
        yp_bas = baseline_preds[idx]
        yp_ful = full_preds[idx]

        f2_bas = fbeta_score(y_b, yp_bas, beta=2, zero_division=0)
        f2_ful = fbeta_score(y_b, yp_ful, beta=2, zero_division=0)
        delta_boot[i] = f2_ful - f2_bas

    alpha = 1 - confidence_level
    lo = float(np.percentile(delta_boot, 100 * alpha / 2))
    hi = float(np.percentile(delta_boot, 100 * (1 - alpha / 2)))
    point_estimate = float(np.median(delta_boot))

    print(
        f"Bootstrap Delta F2 ({int(confidence_level * 100)}% CI): "
        f"{point_estimate:+.4f}  [{lo:+.4f}, {hi:+.4f}]"
    )
    if lo > 0:
        print("-> CI entirely above 0: full model is significantly better.")
    elif hi < 0:
        print("-> CI entirely below 0: baseline model is significantly better.")
    else:
        print("-> CI straddles 0: no significant difference detected.")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(delta_boot, bins=60, color="#5B8DB8", alpha=0.75, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, label="No difference")
    ax.axvline(lo, color="#E07B39", linestyle=":", label=f"{int(confidence_level * 100)}% CI")
    ax.axvline(hi, color="#E07B39", linestyle=":")
    ax.axvline(point_estimate, color="#C03B2B", linewidth=1.8, label="Median Delta F2")
    ax.set_xlabel("Delta F2 (full - baseline)")
    ax.set_title("Bootstrap Distribution of F2 Improvement")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "bootstrap_delta_f2.png", dpi=150)
    plt.close()

    return {
        "point": point_estimate,
        "lo": lo,
        "hi": hi,
        "samples": delta_boot,
    }


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


def run_group_ablation(
    df: pd.DataFrame,
    full_features: list[str],
    n_splits: int = 5,
    xgb_params: dict | None = None,
    random_state: int = 42,
    n_bootstrap: int = 2000,
    output_dir: str | Path = "outputs",
    full_results: dict | None = None,
) -> pd.DataFrame:
    """
    Quantify the incremental value of each physical driver group.

    For each group g, retrain on (all features − g) and measure the drop in
    PR-AUC relative to the full model on the same out-of-fold predictions:

        incremental(g) = PR-AUC(all) − PR-AUC(all − g)

    A paired bootstrap 95% CI entirely above zero means group g contributes
    skill that no other group supplies — i.e. that driver carries information
    not already present in the rest of the feature set.
    """
    from src.features import build_feature_groups
    from src.models import run_cross_validation

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = build_feature_groups(full_features)

    print("\n" + "=" * 60)
    print("DRIVER ABLATION — incremental PR-AUC by feature group")
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
        rows.append(
            {
                "group": gname,
                "n_features": len(gfeats),
                "incremental_pr_auc": cmp["delta"],
                "ci_lo": cmp["lo"],
                "ci_hi": cmp["hi"],
                "significant": cmp["lo"] > 0,
            }
        )
        print(
            f"  {gname:<20} ΔPR-AUC={cmp['delta']:+.4f} "
            f"[{cmp['lo']:+.4f}, {cmp['hi']:+.4f}]"
            f"{'  *' if cmp['lo'] > 0 else ''}"
        )

    table = pd.DataFrame(rows).sort_values(
        "incremental_pr_auc", ascending=False
    ).reset_index(drop=True)

    # Horizontal bar chart with 95% CI whiskers.
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(table) + 1.5))
    y = np.arange(len(table))[::-1]
    colors = ["#2E7D32" if s else "#9E9E9E" for s in table["significant"]]
    err_lo = (table["incremental_pr_auc"] - table["ci_lo"]).clip(lower=0)
    err_hi = (table["ci_hi"] - table["incremental_pr_auc"]).clip(lower=0)
    ax.barh(y, table["incremental_pr_auc"], color=colors,
            xerr=[err_lo, err_hi], capsize=3, alpha=0.9)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(table["group"])
    ax.set_xlabel("Incremental PR-AUC (full − without group)")
    ax.set_title("Which drivers matter for 24-h dust onset?\n"
                 "(green = 95% CI above zero)")
    plt.tight_layout()
    plt.savefig(output_dir / "driver_ablation.png", dpi=150)
    plt.close()

    table.to_csv(output_dir / "driver_ablation.csv", index=False)
    return table


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
) -> None:
    """Write the markdown results report for the driver study."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = len(df)
    n_positive = int(df[TARGET].sum())
    pos_rate = 100 * n_positive / n_samples if n_samples else 0
    n_folds = len(model_results["fold_f2"])

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
        "ROC-AUC and F2 (at a per-fold tuned threshold) are reported alongside.",
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

    # Driver ablation — the headline analysis
    sig = ablation_df[ablation_df["significant"]]
    lines.extend(
        [
            "## Driver Ablation",
            "",
            "Incremental skill of each physical driver group: the change in "
            "PR-AUC when that group is removed and the model retrained "
            "(model − without-group), with paired bootstrap 95% CIs on the "
            "out-of-fold predictions. A CI entirely above zero marks a driver "
            "that carries information not already present in the other features.",
            "",
            "| Driver group | # feats | Incremental PR-AUC | 95% CI | Significant |",
            "|--------------|---------|--------------------|--------|-------------|",
        ]
    )
    for _, r in ablation_df.iterrows():
        lines.append(
            f"| {r['group']} | {int(r['n_features'])} | "
            f"{r['incremental_pr_auc']:+.4f} | "
            f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | "
            f"{'**yes**' if r['significant'] else 'no'} |"
        )
    if not sig.empty:
        drivers = ", ".join(sig["group"].tolist())
        lines += ["", f"**Driver groups with significant incremental skill:** {drivers}."]
    else:
        lines += [
            "",
            "No single driver group shows statistically significant incremental "
            "skill at this sample size.",
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
            "- `driver_ablation.png` — incremental PR-AUC by driver group",
            "- `pr_curve.png` — precision–recall curve for the forecast model",
            "- `shap_importance.png` — SHAP feature importance",
            "",
            "## Conclusion",
            "",
        ]
    )

    if not sig.empty:
        top = sig.iloc[0]
        intro = (
            f"The forecaster attains a cross-validated PR-AUC of "
            f"{model_results['mean_ap']:.3f} (ROC-AUC {model_results['mean_roc']:.3f}) "
            f"at a {pos_rate:.1f}% base rate. "
        )
        if len(sig) == 1:
            detail = (
                f"The driver ablation identifies **{top['group']}** as the only "
                f"feature group contributing statistically significant incremental "
                f"skill (ΔPR-AUC {top['incremental_pr_auc']:+.4f}, 95% CI "
                f"[{top['ci_lo']:+.4f}, {top['ci_hi']:+.4f}]). "
            )
        else:
            drivers = ", ".join(sig["group"].tolist())
            detail = (
                f"The driver ablation identifies **{drivers}** as the feature "
                f"groups contributing statistically significant incremental skill, "
                f"with **{top['group']}** the strongest (ΔPR-AUC "
                f"{top['incremental_pr_auc']:+.4f}, 95% CI [{top['ci_lo']:+.4f}, "
                f"{top['ci_hi']:+.4f}]). "
            )
        lines.append(
            intro + detail + "Remaining groups carry information already present "
            "elsewhere in the feature set."
        )
    else:
        lines.append(
            f"The forecaster attains a cross-validated PR-AUC of "
            f"{model_results['mean_ap']:.3f} (ROC-AUC {model_results['mean_roc']:.3f}). "
            "No individual driver group shows significant incremental skill at "
            "this sample size; performance derives from the combined feature set."
        )

    output_path.write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {output_path}")
