"""Gated 'enough evidence?' check. Does not call forecast or ranking math."""

from __future__ import annotations

from src.agent.state import TourismState

MAX_CRITIC_RETRIES = 2


def destination_looks_junk(dest: str | None) -> bool:
    if not dest:
        return False
    d = dest.lower()
    if "instead" in d or d.startswith("should ") or d.startswith("what "):
        return True
    if len(dest.split()) > 6:
        return True
    return False


def critic_has_gap(state: TourismState) -> bool:
    intent = state.get("intent") or {}
    dest = state.get("destination")
    ranked = state.get("ranked_alternatives") or []
    trace = state.get("tool_trace") or []
    if destination_looks_junk(dest):
        return True
    if not dest and (intent.get("needs_prediction") or intent.get("needs_trip_plan") or intent.get("needs_alternatives")):
        return True
    if intent.get("needs_alternatives") and dest and not ranked and "web_search" not in trace:
        return True
    if intent.get("needs_trip_plan") and dest and not (state.get("trip_plan") or []) and "optimize_trip" not in trace:
        return True
    pressure = state.get("period_pressure") or {}
    if state.get("start_date") and not pressure.get("found") and "assess_period_pressure" not in trace:
        return True
    from src.agent.research import nearby_from_research

    if dest and nearby_from_research(state).get("suggest_nearby"):
        if not ranked and "rank_alternatives" not in trace:
            return True
    return False


def critic_patches(state: TourismState) -> dict:
    """Extra intent flags / destination fix for one more planner pass."""
    from src.agent.intent import resolve_intent

    intent = dict(state.get("intent") or {})
    request = state.get("user_request") or ""
    dest = state.get("destination")
    patches: dict = {}
    if destination_looks_junk(dest) or not dest:
        refreshed = resolve_intent(request, use_llm=True)
        new_dest = refreshed.get("destination")
        if new_dest and not destination_looks_junk(new_dest):
            patches["_destination"] = new_dest
            dest = new_dest
        for key in (
            "needs_prediction",
            "needs_alternatives",
            "needs_current_context",
            "needs_trip_plan",
        ):
            if (refreshed.get("intent") or {}).get(key):
                patches[key] = True
        if refreshed.get("interests"):
            patches["_interests"] = list(
                dict.fromkeys(list(state.get("interests") or []) + list(refreshed["interests"]))
            )
        if refreshed.get("must_visit"):
            patches["_must_visit"] = list(
                dict.fromkeys(list(state.get("must_visit") or []) + list(refreshed["must_visit"]))
            )
        if refreshed.get("start_date") and not state.get("start_date"):
            patches["_start_date"] = refreshed["start_date"]
        if refreshed.get("end_date") and not state.get("end_date"):
            patches["_end_date"] = refreshed["end_date"]

    ranked = state.get("ranked_alternatives") or []
    trace = state.get("tool_trace") or []
    if (intent.get("needs_alternatives") or patches.get("needs_alternatives")) and dest and not ranked:
        if "web_search" not in trace:
            patches["needs_current_context"] = True
            patches["needs_alternatives"] = True
    return patches
