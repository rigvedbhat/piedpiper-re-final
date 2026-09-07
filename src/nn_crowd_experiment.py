"""One-shot MLP vs persistence. Does not change the live forecast path."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from src.feature_engineering import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COL,
    build_forecast_table,
    chronological_masks,
)
from src.preprocessing import build_processed_panel
from src.train import make_preprocessor, metrics_dict

REPORTS = Path("reports")


def main() -> None:
    table = build_forecast_table(build_processed_panel())
    masks = chronological_masks(table)
    train, val, test = masks["train"], masks["val"], masks["test"]
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    mlp = Pipeline(
        [
            ("prep", make_preprocessor()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(32, 16),
                    activation="relu",
                    max_iter=800,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.15,
                ),
            ),
        ]
    )
    mlp.fit(table.loc[train, cols], table.loc[train, TARGET_COL])
    persist = table[TARGET_COL].copy() * np.nan
    persist.loc[val] = table.loc[val, "lag_1_total"]
    persist.loc[test] = table.loc[test, "lag_1_total"]
    pred_val = mlp.predict(table.loc[val, cols])
    pred_test = mlp.predict(table.loc[test, cols])
    out = {
        "note": "MLP is experimental only. Live agent still uses persistence.",
        "mlp_val": metrics_dict(table.loc[val, TARGET_COL], pred_val),
        "mlp_test": metrics_dict(table.loc[test, TARGET_COL], pred_test),
        "persistence_val": metrics_dict(table.loc[val, TARGET_COL], table.loc[val, "lag_1_total"]),
        "persistence_test": metrics_dict(table.loc[test, TARGET_COL], table.loc[test, "lag_1_total"]),
    }
    taj = table[(table["monument"] == "Taj Mahal") & (table["period"] == "2024-25")]
    if len(taj):
        row = taj.iloc[0]
        pred = float(mlp.predict(taj[cols])[0])
        out["taj_2024_25"] = {
            "actual": float(row[TARGET_COL]),
            "mlp": pred,
            "persistence": float(row["lag_1_total"]),
        }
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / "nn_vs_persistence.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
