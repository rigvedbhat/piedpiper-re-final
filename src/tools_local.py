"""Local destination, history, and validated persistence forecast tools."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.crowd_classifier import DISCLAIMER, classify_relative_crowd
from src.predict import (
    NEXT_PERIOD,
    forecast_demand,
    load_panel,
    normalize_period,
    resolve_monument,
    valid_history,
)

KAGGLE_PATH = Path("data/raw/indian_tourism_kaggle_travel_data.csv")
MAPPING_PATH = Path("destination_mapping.csv")

INTEREST_TYPE_HINTS = {
    "temple": ["Temple", "Religious", "Pilgrimage"],
    "spiritual": ["Temple", "Religious", "Pilgrimage"],
    "fort": ["Fort", "Palace", "Heritage"],
    "heritage": ["Fort", "Palace", "Heritage", "Monument", "Historical"],
    "nature": ["Hill", "Lake", "Wildlife", "Beach", "National Park", "Waterfall"],
    "adventure": ["Trek", "Adventure", "Wildlife"],
}


@lru_cache(maxsize=1)
def kaggle_catalog() -> pd.DataFrame:
    df = pd.read_csv(KAGGLE_PATH)
    rows = []
    for name, grp in df.groupby("Place_Name", dropna=True):
        rows.append(
            {
                "name": name,
                "source": "kaggle",
                "state": _mode(grp["Location_State"]),
                "city": _mode(grp["Location_City"]),
                "zone": _mode(grp["Zone"]),
                "place_type": _mode(grp["Place_Type"]),
                "significance": _mode(grp["Significance"]),
                "season": _mode(grp["Season"]),
                "google_rating": float(grp["Google_Rating"].median()),
                "ticket_price": float(grp["Ticket_Price"].median()),
                "airport_within_50km": _mode(grp["Airport_Within_50km"]),
            }
        )
    return pd.DataFrame(rows)


def _mode(series: pd.Series) -> str:
    s = series.dropna()
    if s.empty:
        return ""
    return str(s.mode().iloc[0])


@lru_cache(maxsize=1)
def mapping_table() -> pd.DataFrame:
    if not MAPPING_PATH.exists():
        return pd.DataFrame(columns=["official_name", "kaggle_name"])
    return pd.read_csv(MAPPING_PATH)


def search_destinations(
    query: str,
    location: str | None = None,
    interests: list[str] | None = None,
    limit: int = 8,
) -> list[dict]:
    """Search ASI names and Kaggle place metadata."""
    q = (query or "").strip().lower()
    loc = (location or "").strip().lower()
    interest_types = set()
    for item in interests or []:
        for key, types in INTEREST_TYPE_HINTS.items():
            if key in item.lower():
                interest_types.update(t.lower() for t in types)

    hits: list[dict] = []
    panel = load_panel()
    for name in panel["monument"].unique():
        blob = name.lower()
        circle = str(panel.loc[panel["monument"] == name, "asi_circle"].iloc[0])
        score = 0.0
        if q and (q in blob or q in circle.lower()):
            score += 3.0
        if loc and loc in blob + " " + circle.lower():
            score += 1.5
        if score > 0:
            hits.append(
                {
                    "name": name,
                    "source": "asi",
                    "asi_circle": circle,
                    "score": score,
                    "has_asi_history": True,
                }
            )

    cat = kaggle_catalog()
    for _, row in cat.iterrows():
        blob = " ".join(
            str(row[c]) for c in ["name", "state", "city", "place_type", "significance"]
        ).lower()
        score = 0.0
        if q and q in blob:
            score += 2.0
        if loc and loc in blob:
            score += 1.5
        if interest_types and str(row["place_type"]).lower() in interest_types:
            score += 1.0
        if score > 0:
            hits.append(
                {
                    "name": row["name"],
                    "source": "kaggle",
                    "state": row["state"],
                    "place_type": row["place_type"],
                    "zone": row["zone"],
                    "score": score,
                    "has_asi_history": False,
                }
            )

    hits.sort(key=lambda r: r["score"], reverse=True)
    seen = set()
    out = []
    for h in hits:
        key = (h["name"].lower(), h["source"])
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
        if len(out) >= limit:
            break
    return out


def get_destination_profile(destination: str) -> dict:
    """Structured metadata from Kaggle and/or ASI, if present."""
    cat = kaggle_catalog()
    kaggle_row = cat[cat["name"].str.lower() == destination.strip().lower()]
    mapping = mapping_table()
    profile = {
        "query": destination,
        "asi_name": None,
        "kaggle_name": None,
        "has_asi_history": False,
        "kaggle": None,
        "asi_circle": None,
    }
    panel = load_panel()
    try:
        asi_name = resolve_monument(panel, destination)
        profile["asi_name"] = asi_name
        profile["has_asi_history"] = True
        profile["asi_circle"] = str(
            panel.loc[panel["monument"] == asi_name, "asi_circle"].iloc[0]
        )
    except ValueError:
        asi_name = None

    kaggle_name = None
    if not kaggle_row.empty:
        kaggle_name = str(kaggle_row.iloc[0]["name"])
    elif asi_name is not None and not mapping.empty:
        m = mapping[mapping["official_name"] == asi_name]
        if not m.empty and isinstance(m.iloc[0].get("kaggle_name"), str):
            kn = str(m.iloc[0]["kaggle_name"]).strip()
            if kn:
                kaggle_name = kn
                kaggle_row = cat[cat["name"] == kn]

    profile["kaggle_name"] = kaggle_name
    if kaggle_name and not kaggle_row.empty:
        r = kaggle_row.iloc[0]
        profile["kaggle"] = {
            "place_type": r["place_type"],
            "state": r["state"],
            "city": r["city"],
            "zone": r["zone"],
            "significance": r["significance"],
            "season": r["season"],
            "google_rating": r["google_rating"],
            "ticket_price": r["ticket_price"],
            "airport_within_50km": r["airport_within_50km"],
        }
    elif kaggle_name:
        match = cat[cat["name"] == kaggle_name]
        if not match.empty:
            r = match.iloc[0]
            profile["kaggle"] = {
                "place_type": r["place_type"],
                "state": r["state"],
                "city": r["city"],
                "zone": r["zone"],
                "significance": r["significance"],
                "season": r["season"],
                "google_rating": r["google_rating"],
                "ticket_price": r["ticket_price"],
                "airport_within_50km": r["airport_within_50km"],
            }
    return profile


def _annual_series_direction(observations: list[dict]) -> str:
    totals = [o.get("total_visitors") for o in observations if o.get("total_visitors") is not None]
    if len(totals) < 2:
        return "unknown"
    prev, last = float(totals[-2]), float(totals[-1])
    if last > prev * 1.05:
        return "increasing"
    if last < prev * 0.95:
        return "decreasing"
    return "similar"


def get_historical_footfall(destination: str) -> dict:
    panel = load_panel()
    try:
        name = resolve_monument(panel, destination)
    except ValueError:
        return {
            "destination": destination,
            "found": False,
            "granularity": "unknown",
            "observations": [],
            "note": "No ASI monument-level annual series for this name.",
        }
    hist = valid_history(panel, name)
    observations = [
        {
            "period": str(r["period"]),
            "domestic": None if pd.isna(r["domestic_visitors"]) else float(r["domestic_visitors"]),
            "foreign": None if pd.isna(r["foreign_visitors"]) else float(r["foreign_visitors"]),
            "total_visitors": float(r["total_visitors"]),
        }
        for _, r in hist.iterrows()
    ]
    return {
        "destination": name,
        "found": True,
        "granularity": "annual",
        "annual_direction": _annual_series_direction(observations),
        "asi_circle": str(hist.iloc[-1]["asi_circle"]) if len(hist) else None,
        "observations": observations,
        "note": "Official ASI ticketed monument visitors by financial year.",
    }


def forecast_crowd(destination: str, forecast_period: str | None = None) -> dict:
    """Deterministic persistence forecast. LLM must not compute this."""
    period = normalize_period(forecast_period)
    panel = load_panel()
    try:
        name = resolve_monument(panel, destination)
    except ValueError as exc:
        return {
            "destination": destination,
            "found": False,
            "forecast_period": period,
            "predicted_visitors": None,
            "forecast_method": "unavailable",
            "data_granularity": "unknown",
            "layer": "MODEL OUTPUT",
            "note": str(exc),
        }
    raw = forecast_demand(panel, name, period)
    return {
        "destination": raw["monument"],
        "forecast_period": raw["forecast_period"],
        "predicted_visitors": raw["predicted_visitors"],
        "forecast_method": "persistence",
        "data_granularity": "annual",
        "lag_period": raw["lag_period"],
        "trend": raw["trend"],
        "asi_circle": raw["asi_circle"],
        "layer": "MODEL OUTPUT",
        "found": True,
    }


def classify_crowd(destination: str, predicted_visitors: float | None = None) -> dict:
    panel = load_panel()
    name = resolve_monument(panel, destination)
    if predicted_visitors is None:
        predicted_visitors = forecast_crowd(name)["predicted_visitors"]
    hist = valid_history(panel, name)["total_visitors"].to_numpy()
    result = classify_relative_crowd(float(predicted_visitors), hist)
    result["destination"] = name
    result["predicted_visitors"] = float(predicted_visitors)
    result["layer"] = "MODEL OUTPUT"
    result["disclaimer"] = DISCLAIMER
    return result
