"""Tests for the methodology additions: BH-FDR, naive baselines, seed
robustness, and calibration plotting."""

import numpy as np
import pandas as pd

from src.evaluation import (
    benjamini_hochberg,
    compute_naive_baselines,
    plot_calibration,
    seed_robustness,
)


class TestBenjaminiHochberg:
    def test_known_two_value_case(self):
        # m=2: sorted [0.001, 0.5] -> adj [0.002, 0.5]; reject [True, False]
        reject, adj = benjamini_hochberg(np.array([0.001, 0.5]), q=0.05)
        assert np.allclose(adj, [0.002, 0.5])
        assert reject.tolist() == [True, False]

    def test_matches_report_example(self):
        # The exact real-data ablation p-values; only the smallest survives.
        p = np.array([0.003, 0.175, 0.166, 0.135, 0.640,
                      0.781, 0.585, 0.809, 0.814, 0.651])
        reject, adj = benjamini_hochberg(p, q=0.05)
        assert np.isclose(adj[0], 0.030, atol=1e-3)   # vegetation -> 0.03
        assert reject.sum() == 1 and reject[0]
        assert (adj >= 0).all() and (adj <= 1).all()

    def test_all_null_rejects_none(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.2, 1.0, 20)  # no true effects
        reject, adj = benjamini_hochberg(p, q=0.05)
        assert reject.sum() == 0

    def test_order_preserved(self):
        p = np.array([0.5, 0.01, 0.2])
        _, adj = benjamini_hochberg(p, q=0.05)
        # Adjusted p for the smallest raw p is the smallest adjusted p.
        assert np.argmin(adj) == np.argmin(p)


def _cv_frame(n_days=400, stations=("a", "b"), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_days, freq="D")
    frames = []
    for s in stations:
        sig = rng.normal(0, 1, n_days)
        y = (sig + rng.normal(0, 0.5, n_days) > 1.0).astype(int)
        frames.append(pd.DataFrame({
            "date": dates, "station": s,
            "ws_max": sig, "rh_mean": rng.normal(0, 1, n_days),
            "ndvi": rng.normal(0, 1, n_days),
            "dust_event_next_day": y,
        }))
    return pd.concat(frames, ignore_index=True)


class TestNaiveBaselines:
    def test_rows_and_ordering(self):
        df = _cv_frame()
        feats = ["ws_max", "rh_mean", "ndvi"]
        model_results = {"mean_ap": 0.4, "mean_roc": 0.8}
        out = compute_naive_baselines(
            df, feats, model_results, n_splits=3,
            xgb_params={"n_estimators": 40},
        )
        names = out["baseline"].tolist()
        assert names == ["no-skill (base rate)", "persistence",
                         "meteorology-only model", "full model"]
        base_rate = df["dust_event_next_day"].mean()
        assert np.isclose(
            out.loc[out["baseline"] == "no-skill (base rate)", "pr_auc"].iloc[0],
            base_rate,
        )
        # No-skill ROC-AUC is exactly 0.5.
        assert out.loc[0, "roc_auc"] == 0.5
        # Full-model row reflects the supplied model_results.
        assert out.loc[out["baseline"] == "full model", "pr_auc"].iloc[0] == 0.4


class TestSeedRobustness:
    def test_keys_and_nonneg_sd(self):
        df = _cv_frame()
        out = seed_robustness(
            df, ["ws_max", "rh_mean", "ndvi"], ["ws_max"],
            seeds=[1, 2, 3], n_splits=3, xgb_params={"n_estimators": 40},
        )
        for k in ("ap_mean", "ap_sd", "roc_mean", "roc_sd",
                  "top_delta_mean", "top_delta_sd"):
            assert k in out
        assert out["ap_sd"] >= 0 and out["roc_sd"] >= 0
        assert 0.0 <= out["ap_mean"] <= 1.0


class TestCalibrationPlot:
    def test_creates_file(self, tmp_path):
        rng = np.random.default_rng(0)
        y = (rng.random(600) < 0.15).astype(int)
        proba = np.clip(0.15 + 0.5 * y + rng.normal(0, 0.2, 600), 0, 1)
        results = {"fold_true": [y], "fold_proba": [proba]}
        plot_calibration(results, output_dir=tmp_path)
        assert (tmp_path / "calibration.png").exists()
