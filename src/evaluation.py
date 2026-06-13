"""
Model evaluation, statistical comparison, and visualisation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import wilcoxon
from sklearn.metrics import fbeta_score

from src.features import BASELINE_FEATURES, FULL_FEATURES
from src.models import TARGET, run_cross_validation


def concat_cv_results(results: dict) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate true labels and predictions across all CV folds."""
    y_true = np.concatenate(results["fold_true"])
    y_pred = np.concatenate(results["fold_preds"])
    return y_true, y_pred


def per_station_f2(results: dict) -> pd.DataFrame:
    """F2, precision and recall per station over all out-of-fold predictions."""
    from sklearn.metrics import precision_score, recall_score

    if "fold_station" not in results:
        return pd.DataFrame()

    y_true = np.concatenate(results["fold_true"])
    y_pred = np.concatenate(results["fold_preds"])
    stations = np.concatenate(results["fold_station"])

    rows = []
    for st in sorted(np.unique(stations)):
        m = stations == st
        rows.append(
            {
                "station": st,
                "n": int(m.sum()),
                "positives": int(y_true[m].sum()),
                "f2": fbeta_score(y_true[m], y_pred[m], beta=2, zero_division=0),
                "precision": precision_score(y_true[m], y_pred[m], zero_division=0),
                "recall": recall_score(y_true[m], y_pred[m], zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def evaluate_both_models(
    df: pd.DataFrame,
    n_splits: int = 5,
    output_dir: str | Path = "outputs",
    xgb_params: dict | None = None,
    random_state: int = 42,
) -> tuple[dict, dict]:
    """Run CV for baseline and full models; save per-fold comparison chart."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("BASELINE MODEL (no albedo anomaly)")
    print("=" * 55)
    baseline_results = run_cross_validation(
        df,
        BASELINE_FEATURES,
        n_splits=n_splits,
        cv_strategy="time",
        random_state=random_state,
        xgb_params=xgb_params,
    )

    print()
    print("=" * 55)
    print("FULL MODEL (with albedo anomaly)")
    print("=" * 55)
    full_results = run_cross_validation(
        df,
        FULL_FEATURES,
        n_splits=n_splits,
        cv_strategy="time",
        random_state=random_state,
        xgb_params=xgb_params,
    )

    print()
    print("Summary")
    print("-" * 40)
    print(
        f"Baseline mean F2 = {baseline_results['mean_f2']:.4f} "
        f"+/- {baseline_results['std_f2']:.4f}"
    )
    print(
        f"Full     mean F2 = {full_results['mean_f2']:.4f} "
        f"+/- {full_results['std_f2']:.4f}"
    )
    delta = full_results["mean_f2"] - baseline_results["mean_f2"]
    print(f"Delta F2 (full - baseline) = {delta:+.4f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(n_splits)
    w = 0.35
    ax.bar(
        x - w / 2,
        baseline_results["fold_f2"],
        w,
        label="Baseline",
        color="#5B8DB8",
        alpha=0.85,
    )
    ax.bar(
        x + w / 2,
        full_results["fold_f2"],
        w,
        label="Full (+ albedo anom.)",
        color="#E07B39",
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i + 1}" for i in x])
    ax.set_ylabel("F2-score")
    ax.set_title("Per-fold F2: Baseline vs. Full Model")
    ax.legend()
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))
    plt.tight_layout()
    plt.savefig(output_dir / "f2_comparison_by_fold.png", dpi=150)
    plt.close()

    return baseline_results, full_results


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


def full_statistical_analysis(
    baseline_results: dict,
    full_results: dict,
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95,
    random_state: int = 42,
    output_dir: str | Path = "outputs",
) -> dict[str, Any]:
    """Run Wilcoxon test and bootstrap CI; return summary dict."""
    print("\n" + "=" * 60)
    print("STATISTICAL COMPARISON SUMMARY")
    print("=" * 60)

    wtest = wilcoxon_test(baseline_results, full_results)

    y_true_bas, y_pred_bas = concat_cv_results(baseline_results)
    y_true_ful, y_pred_ful = concat_cv_results(full_results)
    assert np.array_equal(y_true_bas, y_true_ful), "True labels mismatch!"

    boot = bootstrap_f2_ci(
        y_true_bas,
        y_pred_bas,
        y_pred_ful,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
        output_dir=output_dir,
    )

    print("\n" + "-" * 60)
    print(f"{'Metric':<35} {'Baseline':>10} {'Full':>10} {'Delta':>10}")
    print("-" * 60)
    print(
        f"{'Mean F2 (CV)':<35} "
        f"{baseline_results['mean_f2']:>10.4f} "
        f"{full_results['mean_f2']:>10.4f} "
        f"{full_results['mean_f2'] - baseline_results['mean_f2']:>+10.4f}"
    )
    print(
        f"{'Wilcoxon p-value':<35} {'—':>10} {'—':>10} {wtest['p']:>10.4f}"
    )
    print(
        f"{'Bootstrap CI lower':<35} {'—':>10} {'—':>10} {boot['lo']:>+10.4f}"
    )
    print(
        f"{'Bootstrap CI upper':<35} {'—':>10} {'—':>10} {boot['hi']:>+10.4f}"
    )
    print("-" * 60)

    return {
        "wilcoxon": wtest,
        "bootstrap": boot,
        "baseline_mean_f2": baseline_results["mean_f2"],
        "full_mean_f2": full_results["mean_f2"],
        "delta_mean_f2": full_results["mean_f2"] - baseline_results["mean_f2"],
    }


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
    stats: dict,
    baseline_results: dict,
    full_results: dict,
    df: pd.DataFrame,
    output_path: str | Path = "results/report.md",
    data_mode: str = "synthetic",
) -> None:
    """Write final markdown report with all results."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = len(df)
    n_positive = int(df[TARGET].sum())
    pos_rate = 100 * n_positive / n_samples if n_samples else 0

    w = stats["wilcoxon"]
    b = stats["bootstrap"]

    lines = [
        "# Dust-Storm Onset Prediction — Results Report",
        "",
        f"**Data mode:** {data_mode}",
        f"**Generated:** pipeline run",
        "",
        "## Dataset Summary",
        "",
        f"- Total samples: {n_samples:,}",
        f"- Positive (dust event next day): {n_positive:,} ({pos_rate:.1f}%)",
        f"- Stations: {', '.join(sorted(df['station'].unique()))}",
        f"- Study period: {df['date'].min()} to {df['date'].max()}",
        "",
        f"## Cross-Validation Results (TimeSeriesSplit, {len(full_results['fold_f2'])} folds)",
        "",
        "### Baseline Model (ERA5 + soil + NDVI, no albedo)",
        "",
        "| Fold | F2-score |",
        "|------|----------|",
    ]
    for i, f2 in enumerate(baseline_results["fold_f2"], 1):
        lines.append(f"| {i} | {f2:.4f} |")
    lines.extend(
        [
            f"| **Mean** | **{baseline_results['mean_f2']:.4f}** |",
            f"| Std | {baseline_results['std_f2']:.4f} |",
            "",
            "### Full Model (+ MODIS albedo anomaly within 200 km)",
            "",
            "| Fold | F2-score |",
            "|------|----------|",
        ]
    )
    for i, f2 in enumerate(full_results["fold_f2"], 1):
        lines.append(f"| {i} | {f2:.4f} |")
    lines.extend(
        [
            f"| **Mean** | **{full_results['mean_f2']:.4f}** |",
            f"| Std | {full_results['std_f2']:.4f} |",
            "",
            "## Model Comparison",
            "",
            f"| Metric | Baseline | Full | Delta |",
            f"|--------|----------|------|-------|",
            f"| Mean F2 (CV) | {baseline_results['mean_f2']:.4f} | "
            f"{full_results['mean_f2']:.4f} | {stats['delta_mean_f2']:+.4f} |",
            "",
        ]
    )

    # Per-station out-of-fold F2 (baseline vs full)
    base_ps = per_station_f2(baseline_results)
    full_ps = per_station_f2(full_results)
    if not full_ps.empty:
        merged = base_ps.merge(
            full_ps, on=["station", "n", "positives"], suffixes=("_base", "_full")
        )
        lines.extend(
            [
                "## Per-Station Out-of-Fold F2",
                "",
                "| Station | n | Positives | Baseline F2 | Full F2 | Delta |",
                "|---------|---|-----------|-------------|---------|-------|",
            ]
        )
        for _, r in merged.iterrows():
            lines.append(
                f"| {r['station']} | {int(r['n'])} | {int(r['positives'])} | "
                f"{r['f2_base']:.4f} | {r['f2_full']:.4f} | "
                f"{r['f2_full'] - r['f2_base']:+.4f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Statistical Tests",
            "",
            "### Wilcoxon Signed-Rank Test (per-fold F2 differences)",
            "",
            f"- Per-fold differences: {np.round(w['differences'], 4).tolist()}",
            f"- W statistic: {w['W']:.2f}",
            f"- p-value: {w['p']:.4f}",
            f"- Significant at alpha=0.05: {'Yes' if w['p'] < 0.05 else 'No'}",
            "",
            "### Bootstrap Confidence Interval (5000 resamples)",
            "",
            f"- Point estimate (median Delta F2): {b['point']:+.4f}",
            f"- 95% CI: [{b['lo']:+.4f}, {b['hi']:+.4f}]",
        ]
    )
    if b["lo"] > 0:
        lines.append("- **Interpretation:** CI entirely above 0 — full model significantly better.")
    elif b["hi"] < 0:
        lines.append("- **Interpretation:** CI entirely below 0 — baseline significantly better.")
    else:
        lines.append("- **Interpretation:** CI straddles 0 — no significant difference detected.")

    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `outputs/f2_comparison_by_fold.png` — per-fold F2 bar chart",
            "- `outputs/bootstrap_delta_f2.png` — bootstrap Delta F2 distribution",
            "- `outputs/shap_importance.png` — SHAP feature importance (full model)",
            "",
            "## Conclusion",
            "",
        ]
    )

    if stats["delta_mean_f2"] > 0 and b["lo"] > 0:
        lines.append(
            "Adding dynamic MODIS shortwave broadband albedo anomaly features "
            "(within 200 km of each station) improves 24-hour dust-storm onset "
            "prediction F2-score relative to the baseline meteorological model. "
            "Both the mean CV improvement and bootstrap 95% CI support this finding."
        )
    elif stats["delta_mean_f2"] > 0:
        lines.append(
            "The full model shows a positive mean F2 improvement, but the bootstrap "
            "confidence interval includes zero — treat the albedo signal as suggestive "
            "but not conclusively significant with this sample."
        )
    else:
        lines.append(
            "The full model did not outperform the baseline in this run. "
            "With real MODIS/ERA5 data and more dust events, re-evaluate."
        )

    output_path.write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {output_path}")
