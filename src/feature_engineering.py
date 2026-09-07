"""Leakage-safe panel features for next-period visitor demand.

Each row predicts period t using only periods strictly before t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "lag_1_total",
    "lag_2_total",
    "lag_3_total",
    "rolling_mean_2",
    "rolling_mean_3",
    "previous_growth_rate",
    "recent_trend",
    "lag_1_domestic_share",
    "year_index",
    "covid_year",
    "recovery_year",
    "post_covid",
    "n_prior_obs",
    "log_lag_1",
]

CATEGORICAL_FEATURES = ["asi_circle"]
TARGET_COL = "target_total"


def _safe_share(domestic, total) -> float:
    if np.isfinite(domestic) and np.isfinite(total) and total > 0:
        return float(domestic / total)
    return np.nan


def build_forecast_table(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per monument-period where lag_1 exists and target is usable."""
    rows = []
    for monument, grp in panel.sort_values("year_index").groupby("monument", sort=False):
        grp = grp.reset_index(drop=True)
        totals = grp["total_visitors"].to_numpy(dtype=float)
        domestics = grp["domestic_visitors"].to_numpy(dtype=float)
        for i in range(1, len(grp)):
            target = totals[i]
            lag1 = totals[i - 1]
            if not np.isfinite(lag1):
                continue
            rec = grp.iloc[i]
            if rec["exclude_from_training"] or not np.isfinite(target):
                continue

            lag2 = totals[i - 2] if i >= 2 else np.nan
            lag3 = totals[i - 3] if i >= 3 else np.nan
            lag2_filled = lag2 if np.isfinite(lag2) else lag1
            lag3_filled = lag3 if np.isfinite(lag3) else lag2_filled

            prior = [x for x in (lag1, lag2, lag3) if np.isfinite(x)]
            roll2 = float(np.mean(prior[:2]))
            roll3 = float(np.mean(prior[:3])) if prior else lag1

            if np.isfinite(lag2) and lag2 > 0:
                growth = (lag1 - lag2) / lag2
            else:
                growth = 0.0
            trend = (lag1 - lag2_filled) / max(lag2_filled, 1.0)

            rows.append(
                {
                    "s_no": rec["s_no"],
                    "monument": monument,
                    "asi_circle": rec["asi_circle"],
                    "period": rec["period"],
                    "year_index": rec["year_index"],
                    "covid_year": rec["covid_year"],
                    "recovery_year": rec["recovery_year"],
                    "post_covid": rec["post_covid"],
                    "lag_1_total": float(lag1),
                    "lag_2_total": float(lag2_filled),
                    "lag_3_total": float(lag3_filled),
                    "lag_2_was_missing": int(not np.isfinite(lag2)),
                    "lag_3_was_missing": int(not np.isfinite(lag3)),
                    "rolling_mean_2": roll2,
                    "rolling_mean_3": roll3,
                    "previous_growth_rate": float(growth),
                    "recent_trend": float(trend),
                    "lag_1_domestic_share": _safe_share(domestics[i - 1], lag1),
                    "n_prior_obs": len(prior),
                    "log_lag_1": float(np.log1p(lag1)),
                    "target_total": float(target),
                    "baseline_pred": float(lag1),
                }
            )
    out = pd.DataFrame(rows)
    med_share = out["lag_1_domestic_share"].median()
    out["lag_1_domestic_share"] = out["lag_1_domestic_share"].fillna(med_share)
    return out


def chronological_masks(table: pd.DataFrame) -> dict[str, pd.Series]:
    """Train through 2022-23, validate 2023-24, test 2024-25."""
    return {
        "train": table["year_index"] <= 3,
        "val": table["year_index"] == 4,
        "test": table["year_index"] == 5,
    }


def size_bucket(lag_1: pd.Series) -> pd.Series:
    return pd.cut(
        lag_1,
        bins=[-np.inf, 100_000, 1_000_000, np.inf],
        labels=["small", "medium", "large"],
    )
