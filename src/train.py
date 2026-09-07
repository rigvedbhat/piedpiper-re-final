"""Chronological experiment: baseline vs Ridge / Random Forest / Gradient Boosting.

Run from the project root:

    python -m src.train
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_loader import PERIODS
from src.feature_engineering import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COL,
    build_forecast_table,
    chronological_masks,
    size_bucket,
)
from src.preprocessing import build_processed_panel

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
MODELS_DIR = Path("models")
RANDOM_STATE = 42


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def metrics_dict(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n": int(len(y_true)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_pct": mape(y_true, y_pred),
    }


def size_metrics(frame: pd.DataFrame, pred_col: str) -> dict:
    buckets = size_bucket(frame["lag_1_total"])
    out = {}
    for name in ["small", "medium", "large"]:
        sub = frame[buckets == name]
        if len(sub) == 0:
            continue
        out[name] = metrics_dict(sub[TARGET_COL], sub[pred_col])
    return out


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def make_models() -> dict:
    return {
        "Ridge": Pipeline(
            [
                ("prep", make_preprocessor()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("prep", make_preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=6,
                        min_samples_leaf=3,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "GradientBoosting": Pipeline(
            [
                ("prep", make_preprocessor()),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=200,
                        max_depth=3,
                        learning_rate=0.05,
                        min_samples_leaf=3,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def fit_predict(model, train_X, train_y, pred_X):
    model.fit(train_X, train_y)
    return model.predict(pred_X)


def evaluate_split(name: str, frame: pd.DataFrame, pred_col: str) -> dict:
    result = metrics_dict(frame[TARGET_COL], frame[pred_col])
    result["by_size"] = size_metrics(frame, pred_col)
    return result


def walk_forward(table: pd.DataFrame, model_factory) -> list[dict]:
    """Train on all earlier target periods, predict the next period."""
    rows = []
    for holdout_idx in [2, 3, 4, 5]:
        train = table[table["year_index"] < holdout_idx]
        test = table[table["year_index"] == holdout_idx]
        if len(train) == 0 or len(test) == 0:
            continue
        model = model_factory()
        pred = fit_predict(
            model,
            train[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
            np.log1p(train[TARGET_COL]),
            test[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
        )
        pred = np.expm1(np.clip(pred, 0, None))
        m = metrics_dict(test[TARGET_COL], pred)
        m["holdout_period"] = PERIODS[holdout_idx]
        rows.append(m)
    return rows


def run() -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    panel = build_processed_panel()
    panel.to_csv(PROCESSED_DIR / "monument_visitors_long.csv", index=False)

    table = build_forecast_table(panel)
    table.to_csv(PROCESSED_DIR / "forecast_table.csv", index=False)

    masks = chronological_masks(table)
    train = table[masks["train"]].copy()
    val = table[masks["val"]].copy()
    test = table[masks["test"]].copy()

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    y_train_log = np.log1p(train[TARGET_COL])

    summary = {
        "n_monuments_raw": int(panel["monument"].nunique()),
        "n_panel_rows": int(len(panel)),
        "n_usable_forecast_rows": int(len(table)),
        "n_monuments_in_forecast_table": int(table["monument"].nunique()),
        "training_periods": [p for i, p in enumerate(PERIODS) if i <= 3 and i >= 1],
        "validation_period": "2023-24",
        "test_period": "2024-25",
        "features": feature_cols,
        "target": "next_period_total_visitors (log1p trained, original-scale metrics)",
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "n_excluded_panel_rows": int(panel["exclude_from_training"].sum()),
        "splits": {
            "train_year_index": "<= 3 (target periods 2020-21 through 2022-23)",
            "val_year_index": "4 (2023-24)",
            "test_year_index": "5 (2024-25)",
        },
    }

    # Baseline: previous period (lag_1)
    for split_name, frame in [("val", val), ("test", test)]:
        frame = frame.copy()
        frame["pred"] = frame["baseline_pred"]
        summary[f"baseline_{split_name}"] = evaluate_split("baseline", frame, "pred")

    models = make_models()
    val_maes = {}
    fitted = {}
    for name, model in models.items():
        pred_val = np.expm1(
            np.clip(
                fit_predict(
                    model,
                    train[feature_cols],
                    y_train_log,
                    val[feature_cols],
                ),
                0,
                None,
            )
        )
        fitted[name] = model
        val_frame = val.copy()
        val_frame["pred"] = pred_val
        summary[f"{name}_val"] = evaluate_split(name, val_frame, "pred")
        val_maes[name] = summary[f"{name}_val"]["MAE"]

        pred_test = np.expm1(np.clip(model.predict(test[feature_cols]), 0, None))
        test_frame = test.copy()
        test_frame["pred"] = pred_test
        summary[f"{name}_test"] = evaluate_split(name, test_frame, "pred")

    winner = min(val_maes, key=val_maes.get)
    baseline_val_mae = summary["baseline_val"]["MAE"]
    summary["winner_by_val_MAE"] = winner
    summary["ml_beats_baseline_on_val"] = bool(val_maes[winner] < baseline_val_mae)
    summary["ml_beats_baseline_on_test"] = bool(
        summary[f"{winner}_test"]["MAE"] < summary["baseline_test"]["MAE"]
    )

    # COVID sensitivity: drop 2020-21 target rows from training
    train_no_covid = train[train["covid_year"] == 0]
    gb = make_models()["GradientBoosting"]
    pred_val_nc = np.expm1(
        np.clip(
            fit_predict(
                gb,
                train_no_covid[feature_cols],
                np.log1p(train_no_covid[TARGET_COL]),
                val[feature_cols],
            ),
            0,
            None,
        )
    )
    val_nc = val.copy()
    val_nc["pred"] = pred_val_nc
    summary["GradientBoosting_no_covid_targets_val"] = evaluate_split(
        "GB_no_covid", val_nc, "pred"
    )

    def gb_factory():
        return make_models()["GradientBoosting"]

    summary["walk_forward_GradientBoosting"] = walk_forward(table, gb_factory)

    joblib.dump(fitted[winner], MODELS_DIR / "demand_model.joblib")
    joblib.dump(
        {
            "winner": winner,
            "features": feature_cols,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "periods": PERIODS,
        },
        MODELS_DIR / "demand_model_meta.joblib",
    )

    (REPORTS_DIR / "experiment_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("=" * 72)
    print("ASI ANNUAL DEMAND EXPERIMENT")
    print("=" * 72)
    print(f"Monuments (raw):              {summary['n_monuments_raw']}")
    print(f"Panel rows (monument-year):   {summary['n_panel_rows']}")
    print(f"Usable forecast rows:          {summary['n_usable_forecast_rows']}")
    print(f"Monuments in model table:     {summary['n_monuments_in_forecast_table']}")
    print(f"Excluded panel rows:          {summary['n_excluded_panel_rows']}")
    print(f"Train rows / periods:          {summary['n_train']} / {summary['training_periods']}")
    print(f"Validation period:            {summary['validation_period']} (n={summary['n_val']})")
    print(f"Test period:                   {summary['test_period']} (n={summary['n_test']})")
    print(f"Features:                     {feature_cols}")
    print()
    print(f"{'Model':<22} {'Split':<6} {'MAE':>12} {'RMSE':>12} {'R2':>8} {'MAPE%':>8}")
    print("-" * 72)
    for label in ["baseline", "Ridge", "RandomForest", "GradientBoosting"]:
        for split in ["val", "test"]:
            m = summary[f"{label}_{split}"]
            print(
                f"{label:<22} {split:<6} {m['MAE']:12,.0f} {m['RMSE']:12,.0f} "
                f"{m['R2']:8.3f} {m['MAPE_pct']:8.1f}"
            )
    print()
    print(f"Winner (lowest val MAE): {winner}")
    print(f"ML beats baseline on val:  {summary['ml_beats_baseline_on_val']}")
    print(f"ML beats baseline on test: {summary['ml_beats_baseline_on_test']}")
    nc = summary["GradientBoosting_no_covid_targets_val"]
    print(
        "GB val MAE if COVID-year targets dropped from train: "
        f"{nc['MAE']:,.0f} (vs GB-with-COVID {summary['GradientBoosting_val']['MAE']:,.0f})"
    )
    print("Walk-forward GB:")
    for row in summary["walk_forward_GradientBoosting"]:
        print(
            f"  holdout {row['holdout_period']}: MAE={row['MAE']:,.0f}  "
            f"RMSE={row['RMSE']:,.0f}  R2={row['R2']:.3f}"
        )
    print("=" * 72)
    return summary


if __name__ == "__main__":
    run()
