"""Leakage-safe visitor-demand forecast.

Experiments showed last-period persistence beats Ridge/RF/GB on this
annual ASI panel. The live predictor is therefore persistence, not a
tree model that shrinks large sites toward the mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import PERIODS
from src.preprocessing import build_processed_panel

NEXT_PERIOD = "2025-26"
FORECAST_PERIODS = PERIODS + [NEXT_PERIOD]


def load_panel() -> pd.DataFrame:
    return build_processed_panel()


def valid_history(panel: pd.DataFrame, monument: str) -> pd.DataFrame:
    hist = panel[(panel["monument"] == monument) & panel["total_visitors"].notna()].copy()
    hist = hist[~hist["flag_source_anomaly"]]
    return hist.sort_values("year_index")


def resolve_monument(panel: pd.DataFrame, query: str) -> str:
    names = panel["monument"].unique().tolist()
    q = query.strip().lower()
    exact = [n for n in names if n.lower() == q]
    if exact:
        return exact[0]
    contains = [n for n in names if q in n.lower() or n.lower().replace(",", "") in q]
    if len(contains) == 1:
        return contains[0]
    if contains:
        contains.sort(key=lambda n: len(n))
        return contains[0]
    tokens = [t for t in q.replace(",", " ").split() if len(t) >= 4]
    token_hits = [n for n in names if any(t in n.lower() for t in tokens)] if tokens else []
    if len(token_hits) == 1:
        return token_hits[0]
    raise ValueError(
        f"No ASI monument matched {query!r}. Annual forecast is unavailable for this name."
    )


def last_two_totals(hist: pd.DataFrame) -> tuple[float, float | None, str, str | None]:
    hist = hist.sort_values("year_index")
    last = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) >= 2 else None
    return (
        float(last["total_visitors"]),
        float(prev["total_visitors"]) if prev is not None else None,
        str(last["period"]),
        str(prev["period"]) if prev is not None else None,
    )


def trend_label(last: float, prev: float | None) -> str:
    if prev is None or prev <= 0:
        return "Not enough history to describe a trend"
    change = (last - prev) / prev
    if change > 0.05:
        return f"Increasing ({change:+.0%} vs {prev:,.0f})"
    if change < -0.05:
        return f"Decreasing ({change:+.0%} vs {prev:,.0f})"
    return f"Stable ({change:+.0%} vs {prev:,.0f})"


def normalize_period(period: str | None) -> str:
    if not period:
        return NEXT_PERIOD
    p = str(period).strip()
    if p in FORECAST_PERIODS:
        return p
    import re
    m = re.match(r"^(\d{4})-(\d{2,4})$", p)
    if m:
        start_y = m.group(1)
        end_y = m.group(2)
        if len(end_y) == 4:
            end_y = end_y[2:]
        normalized = f"{start_y}-{end_y}"
        if normalized in FORECAST_PERIODS:
            return normalized
    if p in {"2024", "2025", "2026"}:
        return NEXT_PERIOD
    return NEXT_PERIOD


def forecast_demand(panel: pd.DataFrame, monument: str, period: str) -> dict:
    """Forecast annual demand for `period` using last observed FY total."""
    period = normalize_period(period)
    if period not in FORECAST_PERIODS:
        raise ValueError(f"Unsupported period {period}. Choose from {FORECAST_PERIODS}")
    hist = valid_history(panel, monument)
    if hist.empty:
        raise ValueError(f"No usable visitor history for {monument}")

    last, prev, last_period, prev_period = last_two_totals(hist)
    # Persistence: predicted next FY = last observed total.
    # For a backtest year that already exists, still use the prior FY only.
    if period in PERIODS:
        prior = hist[hist["year_index"] < PERIODS.index(period)]
        if prior.empty:
            raise ValueError(f"No prior year available to forecast {period} for {monument}")
        predicted = float(prior.iloc[-1]["total_visitors"])
        used_as_lag = str(prior.iloc[-1]["period"])
        actual = float(hist.loc[hist["period"] == period, "total_visitors"].iloc[0]) if period in set(hist["period"]) else None
    else:
        predicted = last
        used_as_lag = last_period
        actual = None

    circle = str(hist.iloc[-1]["asi_circle"])
    return {
        "monument": monument,
        "asi_circle": circle,
        "forecast_period": period,
        "predicted_visitors": predicted,
        "method": "persistence_last_observed_year",
        "lag_period": used_as_lag,
        "previous_period_visitors": last if period not in PERIODS else last,
        "lag_visitors": predicted if period in PERIODS else last,
        "previous_period": last_period,
        "prior_period_visitors": prev,
        "prior_period": prev_period,
        "trend": trend_label(last, prev),
        "history": hist,
        "actual_if_known": actual,
        "n_history_years": int(len(hist)),
    }
