"""Relative crowd level from a monument's own history. Not occupancy."""

from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = ("LOW", "MODERATE", "HIGH", "VERY HIGH")

DISCLAIMER = (
    "Relative crowd level based on historical visitor demand; not physical occupancy."
)


def classify_relative_crowd(predicted: float, history_totals: np.ndarray) -> dict:
    hist = np.asarray(history_totals, dtype=float)
    hist = hist[np.isfinite(hist)]
    method = "monument_quartiles"
    if len(hist) < 4:
        method = "sparse_history_tertiles_then_high"
        if len(hist) == 0:
            return {
                "level": "UNKNOWN",
                "method": "no_history",
                "disclaimer": DISCLAIMER,
                "cuts": None,
            }
        if len(hist) == 1:
            # One point: predicted vs that single year.
            ratio = predicted / hist[0] if hist[0] else 1.0
            if ratio < 0.85:
                level = "LOW"
            elif ratio < 1.05:
                level = "MODERATE"
            elif ratio < 1.2:
                level = "HIGH"
            else:
                level = "VERY HIGH"
            return {
                "level": level,
                "method": "single_year_ratio",
                "disclaimer": DISCLAIMER,
                "cuts": {"ref": float(hist[0])},
            }
        p33, p67 = np.percentile(hist, [33, 67])
        cuts = {"p33": float(p33), "p67": float(p67)}
        if predicted <= p33:
            level = "LOW"
        elif predicted <= p67:
            level = "MODERATE"
        else:
            level = "HIGH" if predicted <= np.max(hist) else "VERY HIGH"
        return {"level": level, "method": method, "disclaimer": DISCLAIMER, "cuts": cuts}

    p25, p50, p75 = np.percentile(hist, [25, 50, 75])
    if predicted <= p25:
        level = "LOW"
    elif predicted <= p50:
        level = "MODERATE"
    elif predicted <= p75:
        level = "HIGH"
    else:
        level = "VERY HIGH"
    return {
        "level": level,
        "method": method,
        "disclaimer": DISCLAIMER,
        "cuts": {"p25": float(p25), "p50": float(p50), "p75": float(p75)},
    }
