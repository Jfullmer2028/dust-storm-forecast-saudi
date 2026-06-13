"""Tests for evaluation: per-station F2, Wilcoxon test, bootstrap CI."""

import numpy as np

from src.evaluation import (
    bootstrap_auc_ci,
    bootstrap_f2_ci,
    per_station_f2,
    wilcoxon_test,
)


def _fake_results(fold_f2, fold_true, fold_preds, fold_station=None, fold_proba=None):
    res = {
        "fold_f2": np.array(fold_f2),
        "fold_true": fold_true,
        "fold_preds": fold_preds,
    }
    if fold_station is not None:
        res["fold_station"] = fold_station
    if fold_proba is not None:
        res["fold_proba"] = fold_proba
    return res


class TestPerStationF2:
    def test_separates_stations(self):
        results = _fake_results(
            fold_f2=[0.5],
            fold_true=[np.array([1, 1, 0, 0])],
            fold_preds=[np.array([1, 1, 0, 0])],  # perfect
            fold_station=[np.array(["a", "a", "b", "b"])],
            fold_proba=[np.array([0.9, 0.8, 0.1, 0.2])],
        )
        df = per_station_f2(results)
        assert set(df["station"]) == {"a", "b"}
        # Station 'a' has both positives correctly predicted -> F2 = 1
        assert df.loc[df["station"] == "a", "f2"].iloc[0] == 1.0
        assert "ap" in df.columns

    def test_empty_when_no_station_tracking(self):
        results = _fake_results(
            fold_f2=[0.5],
            fold_true=[np.array([1, 0])],
            fold_preds=[np.array([1, 0])],
        )
        assert per_station_f2(results).empty


class TestBootstrapAUC:
    def test_pr_auc_ci_above_zero_when_full_ranks_better(self):
        rng = np.random.default_rng(0)
        n = 3000
        y_true = (rng.random(n) < 0.1).astype(int)  # ~10% positives
        # Full probabilities correlate with truth; baseline barely does.
        proba_full = np.clip(0.1 + 0.7 * y_true + rng.normal(0, 0.15, n), 0, 1)
        proba_base = np.clip(0.1 + 0.1 * y_true + rng.normal(0, 0.3, n), 0, 1)
        out = bootstrap_auc_ci(
            y_true, proba_base, proba_full, metric="ap", n_bootstrap=400
        )
        assert out["full"] > out["baseline"]
        assert out["lo"] <= out["delta"] <= out["hi"]
        assert out["lo"] > 0  # full clearly ranks dust risk better

    def test_roc_auc_metric_runs(self):
        rng = np.random.default_rng(1)
        n = 1000
        y_true = (rng.random(n) < 0.2).astype(int)
        proba = np.clip(0.2 + 0.5 * y_true + rng.normal(0, 0.2, n), 0, 1)
        out = bootstrap_auc_ci(y_true, proba, proba, metric="roc", n_bootstrap=200)
        assert out["metric"] == "ROC-AUC"
        assert abs(out["delta"]) < 1e-9  # identical inputs -> zero delta


class TestWilcoxon:
    def test_consistent_improvement_flagged(self, capsys):
        baseline = _fake_results(
            [0.40, 0.42, 0.38, 0.41, 0.39, 0.43, 0.40, 0.37], [], []
        )
        full = _fake_results(
            [0.50, 0.51, 0.49, 0.52, 0.48, 0.53, 0.50, 0.47], [], []
        )
        out = wilcoxon_test(baseline, full)
        # 8 consistently positive differences -> significant (min p = 2/256)
        assert out["p"] < 0.05


class TestBootstrap:
    def test_ci_above_zero_when_full_better(self, tmp_path):
        rng = np.random.default_rng(0)
        n = 2000
        y_true = rng.integers(0, 2, n)
        # Full predictions match truth more often than baseline
        full_preds = np.where(rng.random(n) < 0.9, y_true, 1 - y_true)
        base_preds = np.where(rng.random(n) < 0.6, y_true, 1 - y_true)
        out = bootstrap_f2_ci(
            y_true, base_preds, full_preds,
            n_bootstrap=500, output_dir=tmp_path,
        )
        assert out["lo"] <= out["point"] <= out["hi"]
        assert out["lo"] > 0  # full clearly better
