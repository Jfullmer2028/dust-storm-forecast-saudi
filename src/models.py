"""
XGBoost classifier training, cross-validation, and hyperparameter tuning.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, TimeSeriesSplit

TARGET = "dust_event_next_day"
GROUP_COL = "station"


def _impute_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Median imputation computed on training fold only."""
    col_medians = np.nanmedian(X_train, axis=0)
    nan_mask_train = np.isnan(X_train)
    nan_mask_test = np.isnan(X_test)
    X_train_imp = np.where(
        nan_mask_train,
        np.tile(col_medians, (X_train.shape[0], 1)),
        X_train,
    )
    X_test_imp = np.where(
        nan_mask_test,
        np.tile(col_medians, (X_test.shape[0], 1)),
        X_test,
    )
    return X_train_imp, X_test_imp


def _make_xgb_classifier(
    scale_pos_weight: float,
    params: dict | None = None,
    random_state: int = 42,
) -> xgb.XGBClassifier:
    defaults = {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": -1,
    }
    if params:
        defaults.update(params)
    return xgb.XGBClassifier(**defaults)


def run_cross_validation(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = 5,
    cv_strategy: str = "time",
    random_state: int = 42,
    xgb_params: dict | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run cross-validation and return per-fold F2 scores, predictions, and labels.

    cv_strategy:
      'time'    — TimeSeriesSplit (expanding window)
      'station' — GroupKFold leave-one-station-out
    """
    # For temporal CV the data must be ordered by date across all stations,
    # otherwise training folds contain dates later than the test fold of
    # another station (temporal leakage).
    sort_cols = ["date", "station"] if cv_strategy == "time" else ["station", "date"]
    df_sorted = df.sort_values(sort_cols).reset_index(drop=True)

    missing = [c for c in feature_cols if c not in df_sorted.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing[:5]}...")

    X = df_sorted[feature_cols].values.astype(float)
    y = df_sorted[TARGET].values.astype(int)
    groups = df_sorted[GROUP_COL].values

    if cv_strategy == "time":
        cv = TimeSeriesSplit(n_splits=n_splits)
        splits = list(cv.split(X))
    elif cv_strategy == "station":
        # True leave-one-station-out: one fold per station.
        n_groups = len(np.unique(groups))
        cv = GroupKFold(n_splits=n_groups)
        splits = list(cv.split(X, y, groups=groups))
    else:
        raise ValueError(f"Unknown cv_strategy: {cv_strategy}")

    fold_f2: list[float] = []
    fold_ap: list[float] = []
    fold_roc: list[float] = []
    fold_preds: list[np.ndarray] = []
    fold_true: list[np.ndarray] = []
    fold_proba: list[np.ndarray] = []
    fold_station: list[np.ndarray] = []
    fold_threshold: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        spw = n_neg / max(n_pos, 1)

        X_train_imp, X_test_imp = _impute_train_test(X_train, X_test)

        # Hold out the most recent slice of the training fold to choose the
        # decision threshold — never the test fold, so there is no leakage.
        val_size = max(int(0.15 * len(X_train_imp)), 1)
        X_tr = X_train_imp[:-val_size]
        X_val = X_train_imp[-val_size:]
        y_tr = y_train[:-val_size]
        y_val = y_train[-val_size:]

        model = _make_xgb_classifier(spw, xgb_params, random_state)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        # Pick the probability threshold that maximises F2 on the validation
        # slice, then apply it to the held-out test fold (rather than a fixed
        # 0.5 cut-off). PR-AUC and ROC-AUC remain threshold-free.
        val_proba = model.predict_proba(X_val)[:, 1]
        threshold = optimal_f2_threshold(y_val, val_proba, beta=2.0)

        y_proba = model.predict_proba(X_test_imp)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
        # Threshold-independent metrics (only defined when the fold has both
        # classes). PR-AUC (average precision) is the primary metric for this
        # rare-event problem; ROC-AUC is reported alongside it.
        has_both = 0 < y_test.sum() < len(y_test)
        ap = average_precision_score(y_test, y_proba) if has_both else float("nan")
        roc = roc_auc_score(y_test, y_proba) if has_both else float("nan")

        fold_f2.append(f2)
        fold_ap.append(ap)
        fold_roc.append(roc)
        fold_preds.append(y_pred)
        fold_true.append(y_test)
        fold_proba.append(y_proba)
        fold_station.append(groups[test_idx])
        fold_threshold.append(threshold)

        if verbose:
            print(
                f"  Fold {fold_idx + 1}: PR-AUC={ap:.4f}  ROC-AUC={roc:.4f}  "
                f"F2={f2:.4f}  "
                f"P={precision_score(y_test, y_pred, zero_division=0):.3f}  "
                f"R={recall_score(y_test, y_pred, zero_division=0):.3f}  "
                f"thr={threshold:.2f}  "
                f"(+:{y_test.sum()}  -:{(y_test == 0).sum()})"
            )

    return {
        "fold_f2": np.array(fold_f2),
        "mean_f2": float(np.mean(fold_f2)),
        "std_f2": float(np.std(fold_f2)),
        "fold_ap": np.array(fold_ap),
        "mean_ap": float(np.nanmean(fold_ap)),
        "std_ap": float(np.nanstd(fold_ap)),
        "fold_roc": np.array(fold_roc),
        "mean_roc": float(np.nanmean(fold_roc)),
        "fold_preds": fold_preds,
        "fold_true": fold_true,
        "fold_proba": fold_proba,
        "fold_station": fold_station,
        "fold_threshold": fold_threshold,
        "feature_cols": feature_cols,
    }


def tune_xgboost(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_trials: int = 60,
    random_state: int = 42,
) -> dict:
    """
    Bayesian hyperparameter search with Optuna, optimising mean F2 over
    TimeSeriesSplit folds.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df_sorted = df.sort_values(["date", "station"]).reset_index(drop=True)
    X = df_sorted[feature_cols].values.astype(float)
    y = df_sorted[TARGET].values.astype(int)
    tscv = TimeSeriesSplit(n_splits=4)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.4, 1.0
            ),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        fold_scores = []
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
            X_tr_imp, X_te_imp = _impute_train_test(X_tr, X_te)

            m = _make_xgb_classifier(spw, params, random_state)
            m.fit(X_tr_imp, y_tr, verbose=False)
            f2 = fbeta_score(y_te, m.predict(X_te_imp), beta=2, zero_division=0)
            fold_scores.append(f2)

        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimal_f2_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    beta: float = 2.0,
) -> float:
    """Search probability threshold that maximises F-beta on validation set."""
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_f2, best_thresh = 0.0, 0.5
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f2 = fbeta_score(y_true, preds, beta=beta, zero_division=0)
        if f2 > best_f2:
            best_f2, best_thresh = f2, t
    return best_thresh
