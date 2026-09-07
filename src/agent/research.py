"""Whether nearby orbits are warranted from the research bundle, not ASI labels alone."""

from __future__ import annotations


def nearby_from_research(state: dict) -> dict:
    """Nearby/underrated places follow the research step, not only annual HIGH/VERY HIGH."""
    reasons: list[str] = []
    pressure = state.get("period_pressure") or {}
    web = state.get("web_context") or {}
    historical = state.get("historical_demand") or {}
    crowds = state.get("crowd_levels") or {}

    if pressure.get("footfall_direction") == "likely_higher":
        reasons.append("travel window (season/weekends/holidays)")
    season = (pressure.get("season") or "")
    if season in {"peak_winter_tourism", "dry_cool_season"} and pressure.get("weekend_days"):
        if "travel window (season/weekends/holidays)" not in reasons:
            reasons.append(f"season={season} with weekend days in the window")
    if pressure.get("holiday_dates"):
        reasons.append("public holiday(s) in the window")
    if web.get("events") or web.get("holidays"):
        reasons.append("events/festivals found in web research")
    if web.get("weather"):
        reasons.append("weather notes from web research")
    if web.get("closures") or web.get("advisories"):
        reasons.append("closures/advisories in web research")

    levels = [(v or {}).get("level") for v in crowds.values()]
    if any(lv in {"HIGH", "VERY HIGH"} for lv in levels):
        reasons.append("annual ASI relative level HIGH/VERY HIGH (one input, not the only gate)")
    for hist in historical.values():
        if (hist or {}).get("found") and (hist or {}).get("annual_direction") == "increasing":
            reasons.append("ASI annual series increasing")
            break

    return {
        "suggest_nearby": bool(reasons),
        "reasons": reasons,
        "note": (
            "Nearby suggestions are based on the research bundle "
            "(window, weather, events, history if any). "
            "Annual HIGH/VERY HIGH is optional evidence, not the sole trigger."
        ),
    }
