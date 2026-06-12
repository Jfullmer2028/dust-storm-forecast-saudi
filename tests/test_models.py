"""Tests for cross-validation correctness (no leakage) and model utilities."""

import numpy as np
import pandas as pd
import pytest

from src.models import (
    TARGET,
    _impute_train_test,
    optimal_f2_threshold,
    run_cross_validation,
)


def _make_cv_frame(n_days=300, stations=("a", "b"), seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    dates = pd.date_range("2018-01-01", periods=n_days, freq="D")
    for s in stations:
        df = pd.DataFrame(
            {
                "date": dates,
                "station": s,
                "x1": rng.random(n_days),
                "x2": rng.random(n_days),
                TARGET: rng.integers(0, 2, n_days),
            }
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


class TestImputation:
    def test_medians_from_train_only(self):
        X_train = np.array([[1.0], [3.0], [np.nan]])
        X_test = np.array([[np.nan], [100.0]])
        X_tr, X_te = _impute_train_test(X_train, X_test)
        assert X_tr[2, 0] == 2.0  # median of train (1, 3)
        assert X_te[0, 0] == 2.0  # test NaN filled with TRAIN median
        assert X_te[1, 0] == 100.0


class TestRunCrossValidation:
    def test_time_cv_has_no_temporal_leakage(self):
        """Every training date must be <= every test date within each fold."""
        df = _make_cv_frame()
        df_sorted = df.sort_values(["date", "station"]).reset_index(drop=True)
        from sklearn.model_selection import TimeSeriesSplit

        for train_idx, test_idx in TimeSeriesSplit(n_splits=3).split(df_sorted):
            assert (
                df_sorted.loc[train_idx, "date"].max()
                <= df_sorted.loc[test_idx, "date"].min()
            )

    def test_returns_expected_keys_and_fold_count(self):
        df = _make_cv_frame()
        results = run_cross_validation(
            df,
            ["x1", "x2"],
            n_splits=3,
            xgb_params={"n_estimators": 10},
            verbose=False,
        )
        assert len(results["fold_f2"]) == 3
        assert {"mean_f2", "std_f2", "fold_preds", "fold_true"} <= set(results)
        assert all(0.0 <= f <= 1.0 for f in results["fold_f2"])

    def test_missing_feature_raises(self):
        df = _make_cv_frame()
        with pytest.raises(ValueError, match="Missing feature columns"):
            run_cross_validation(df, ["nope"], verbose=False)

    def test_station_cv_groups_respected(self):
        df = _make_cv_frame(stations=("a", "b", "c"))
        df_sorted = df.sort_values(["station", "date"]).reset_index(drop=True)
        from sklearn.model_selection import GroupKFold

        groups = df_sorted["station"].values
        for train_idx, test_idx in GroupKFold(n_splits=3).split(
            df_sorted, groups=groups
        ):
            train_stations = set(groups[train_idx])
            test_stations = set(groups[test_idx])
            assert train_stations.isdisjoint(test_stations)


class TestOptimalThreshold:
    def test_perfect_separation(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        t = optimal_f2_threshold(y_true, y_prob)
        assert 0.3 < t <= 0.7
