"""LangChain tool wrappers around existing deterministic Python tools."""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import tool

from src.tools_local import (
    classify_crowd,
    forecast_crowd,
    get_destination_profile,
    get_historical_footfall,
    search_destinations,
)
from src.tools_recommend import find_similar_destinations, rank_alternatives
from src.tools_web import web_search


def _json(data) -> str:
    return json.dumps(data, default=str)


@tool
def clarify_with_user_tool(question: str, missing: str = "") -> str:
    """Ask the traveler a natural follow-up. Use when destination, dates/duration, or party (family/friends/solo) is missing. missing is a comma list such as destination,dates,party. Do not invent research in this turn."""
    return _json({"awaiting_user": True, "question": question, "missing": missing})


@tool
def search_destinations_tool(
    query: str,
    location: str = "",
    interests: str = "",
) -> str:
    """Search local ASI + Kaggle destination names. Use for regions (e.g. Kolhapur, Agra) or attraction names. Interests is a comma-separated list such as temples,forts,nature. Does not forecast crowds."""
    interest_list = [x.strip() for x in interests.split(",") if x.strip()] or None
    return _json(
        search_destinations(
            query,
            location=location or None,
            interests=interest_list,
            limit=8,
        )
    )


@tool
def get_destination_profile_tool(destination: str) -> str:
    """Return structured local metadata (ASI circle, Kaggle type/state if mapped). Does not include live weather or visitor forecasts."""
    return _json(get_destination_profile(destination))


@tool
def get_historical_footfall_tool(destination: str) -> str:
    """Return official ASI annual visitor observations if this name matches a monument. Granularity is annual financial years only. Returns found=false if no ASI series exists."""
    return _json(get_historical_footfall(destination))


@tool
def forecast_crowd_tool(destination: str, forecast_period: str = "2025-26") -> str:
    """Predict next-period visitor demand using validated annual persistence (last observed FY total). data_granularity is always annual. Does NOT predict hourly or daily headcount. Do not invent a different number."""
    return _json(forecast_crowd(destination, forecast_period))


@tool
def classify_crowd_tool(destination: str, predicted_visitors: float | None = None) -> str:
    """Classify relative historical demand as LOW/MODERATE/HIGH/VERY HIGH. Not physical occupancy. Requires an ASI monument with history."""
    return _json(classify_crowd(destination, predicted_visitors))


@tool
def find_similar_destinations_tool(
    destination: str,
    interests: str = "",
    location: str = "",
) -> str:
    """Find relevant alternative destinations (same circle/type/state/interests). Does not rank them. Low popularity alone is not a reason to include a place."""
    interest_list = [x.strip() for x in interests.split(",") if x.strip()] or None
    return _json(
        find_similar_destinations(
            destination,
            interests=interest_list,
            location=location or None,
            limit=12,
        )
    )


@tool
def rank_alternatives_tool(anchor: str, interests: str = "", location: str = "") -> str:
    """Deterministic ranking of alternatives for an anchor. Calls similarity then scoring in Python. Do not invent order or scores. Each item includes final_score, components, predicted_visitors if ASI exists, and why."""
    prefs = {
        "interests": [x.strip() for x in interests.split(",") if x.strip()] or None,
        "location": location or None,
    }
    ranked = rank_alternatives(
        anchor,
        user_preferences=prefs,
    )
    return _json(ranked[:5])


@tool
def web_search_tool(query: str, domains: str = "", official_only: bool = True) -> str:
    """Search the public web for destination facts, famous places, events, weather. You write the query, including official names and aliases (e.g. Belagavi Belgaum). Never fabricate URLs. official_only=true prefers government tourism hosts. Not a visitor-count source."""
    domain_list = [x.strip() for x in domains.split(",") if x.strip()] or None
    try:
        return _json(web_search(query, domains=domain_list, max_results=5))
    except Exception as exc:
        return _json(
            {
                "query": query,
                "provider": "none",
                "results": [],
                "warnings": [f"web_search failed: {exc}"],
                "layer": "FACTS",
            }
        )


@tool
def set_trip_context_tool(
    destination: str,
    also_known_as: str = "",
    state: str = "",
    climate_zone: str = "",
    start_date: str = "",
    end_date: str = "",
    party_type: str = "",
    must_visit: str = "",
    interests: str = "",
    needs_prediction: bool = True,
    needs_alternatives: bool = False,
    needs_trip_plan: bool = False,
    needs_current_context: bool = False,
) -> str:
    """Interpret the trip: official place name, aliases, Indian state, climate_zone, ISO dates, party_type (family|friends|solo|unknown), must-visit, interests, and which research is needed."""
    return _json(
        {
            "destination": destination,
            "also_known_as": also_known_as,
            "state": state,
            "climate_zone": climate_zone,
            "start_date": start_date,
            "end_date": end_date,
            "party_type": party_type,
        }
    )


@tool
def assess_period_pressure_tool(
    destination: str = "",
    start_date: str = "",
    end_date: str = "",
    climate_zone: str = "",
    state: str = "",
) -> str:
    """Heuristic travel-window pressure vs the annual ASI figure (weekends/holidays/season/web). Never a new visitor total. Pass climate_zone/state so February in Karnataka is not treated as north-India peak."""
    return _json({"queued": True, "destination": destination, "start_date": start_date})


@tool
def extract_nearby_places_tool(destination: str, region_state: str = "") -> str:
    """From web hits already retrieved, list attractions in/near the destination only. No crowd numbers. Call after web_search."""
    return _json({"queued": True, "destination": destination})


@tool
def optimize_trip_tool(destination: str = "", start_date: str = "", end_date: str = "") -> str:
    """Build a day sketch from must-visit anchors plus ranked nearby orbits. Not occupancy-optimised."""
    return _json({"queued": True})


ALL_TOOLS = [
    clarify_with_user_tool,
    set_trip_context_tool,
    search_destinations_tool,
    get_destination_profile_tool,
    get_historical_footfall_tool,
    forecast_crowd_tool,
    classify_crowd_tool,
    web_search_tool,
    extract_nearby_places_tool,
    assess_period_pressure_tool,
    find_similar_destinations_tool,
    rank_alternatives_tool,
    optimize_trip_tool,
]
