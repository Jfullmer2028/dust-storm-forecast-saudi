"""Tests for evaluation: per-station F2, bootstrap AUC CI, driver ablation."""

import numpy as np
import pandas as pd

from src.evaluation import (
    bootstrap_auc_ci,
    per_station_f2,
    run_group_ablation,
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


class TestDriverAblation:
    def test_informative_group_ranks_above_noise(self, tmp_path):
        """The group that drives the label must show the largest incremental PR-AUC."""
        rng = np.random.default_rng(0)
        n = 1500
        dates = pd.date_range("2018-01-01", periods=n // 3, freq="D")
        # ws_max carries the signal; soil_clay_0-5cm is pure noise.
        rows = []
        for st in ["a", "b", "c"]:
            signal = rng.normal(0, 1, len(dates))
            y = (signal + rng.normal(0, 0.5, len(dates)) > 1.0).astype(int)
            rows.append(
                pd.DataFrame({
                    "date": dates,
                    "station": st,
                    "ws_max": signal,
                    "rh_mean": rng.normal(0, 1, len(dates)),
                    "soil_clay_0-5cm": rng.normal(0, 1, len(dates)),
                    "dust_event_next_day": y,
                })
            )
        df = pd.concat(rows, ignore_index=True)
        feats = ["ws_max", "rh_mean", "soil_clay_0-5cm"]
        table = run_group_ablation(
            df, feats, n_splits=3, xgb_params={"n_estimators": 60},
            n_bootstrap=300, output_dir=tmp_path,
        )
        assert set(table["group"]) == {"wind_speed", "humidity_dryness", "soil_texture"}
        wind = table.set_index("group").loc["wind_speed", "incremental_pr_auc"]
        soil = table.set_index("group").loc["soil_texture", "incremental_pr_auc"]
        assert wind > soil  # the true driver contributes more than noise
        assert (tmp_path / "driver_ablation.png").exists()

    def test_ablation_has_fdr_columns(self, tmp_path):
        rng = np.random.default_rng(1)
        dates = pd.date_range("2018-01-01", periods=400, freq="D")
        rows = []
        for st in ["a", "b"]:
            sig = rng.normal(0, 1, len(dates))
            y = (sig + rng.normal(0, 0.5, len(dates)) > 1.0).astype(int)
            rows.append(pd.DataFrame({
                "date": dates, "station": st, "ws_max": sig,
                "rh_mean": rng.normal(0, 1, len(dates)),
                "dust_event_next_day": y,
            }))
        df = pd.concat(rows, ignore_index=True)
        table = run_group_ablation(
            df, ["ws_max", "rh_mean"], n_splits=3,
            xgb_params={"n_estimators": 40}, n_bootstrap=300, output_dir=tmp_path,
        )
        for col in ("p_value", "p_fdr", "significant_fdr"):
            assert col in table.columns
        # FDR-adjusted p is never smaller than the raw p.
        assert (table["p_fdr"] >= table["p_value"] - 1e-9).all()
