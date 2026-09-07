"""Traveler-facing explanation of every routing and ranking decision."""

from __future__ import annotations

from src.agent.state import TourismState


def build_decision_reasoning(state: TourismState) -> list[str]:
    intent = state.get("intent") or {}
    dest = state.get("destination") or "the place in your request"
    start, end = state.get("start_date"), state.get("end_date")
    interests = state.get("interests") or []
    must = state.get("must_visit") or []
    forecasts = state.get("forecasts") or {}
    crowds = state.get("crowd_levels") or {}
    ranked = state.get("ranked_alternatives") or []
    plan = state.get("trip_plan") or []
    web = state.get("web_context") or {}
    trace = state.get("tool_trace") or []
    extra = list(state.get("decisions") or [])

    lines: list[str] = []
    n = 1

    def add(title: str, body: str) -> None:
        nonlocal n
        lines.append(f"{n}. {title}")
        lines.append(f"   {body}")
        n += 1

    add(
        "How I read your request",
        f"I treated the destination as {dest}. "
        f"Crowd forecast needed: {bool(intent.get('needs_prediction'))}. "
        f"Alternatives/nearby places needed: {bool(intent.get('needs_alternatives'))}. "
        f"Current web context needed: {bool(intent.get('needs_current_context'))}. "
        f"Trip plan needed: {bool(intent.get('needs_trip_plan'))}. "
        f"Dates: {start or 'not specified'} to {end or 'not specified'}. "
        f"Interests: {', '.join(interests) or 'not specified'}. "
        f"Must-visit: {', '.join(must) or 'none'}.",
    )

    if intent.get("wants_hourly") or state.get("wants_hourly"):
        add(
            "Hour / day request",
            "You mentioned a clock time or a specific day. Our validated model is annual "
            "financial-year persistence only, so I still run that forecast if ASI data exists, "
            "then I explicitly refuse to invent an hourly or occupancy figure.",
        )

    found = [k for k, v in forecasts.items() if v.get("found")]
    missing = [k for k, v in forecasts.items() if not v.get("found")]
    if intent.get("needs_prediction") or predict_was_checked(forecasts, must, dest):
        if found:
            bits = []
            for name in found:
                fc = forecasts[name]
                crowd = crowds.get(fc.get("destination") or name) or {}
                bits.append(
                    f"{fc.get('destination')} → {fc.get('predicted_visitors'):,.0f} visitors "
                    f"({fc.get('forecast_period')}) using last observed ASI year "
                    f"(method={fc.get('forecast_method')}); relative level "
                    f"{crowd.get('level') or 'n/a'} vs this monument's own history, not how full the site is."
                )
            add("Why I produced a crowd number", " ".join(bits))
        if missing:
            add(
                "Why I did not invent crowd numbers",
                "I looked up official ASI monument series for: "
                + ", ".join(missing)
                + ". No validated annual history matched, so predicted_visitors stays empty. "
                "I do not copy words like 'popular' or 'lesser-known' from the web into a count.",
            )
        if not found and not missing and not intent.get("needs_prediction"):
            add(
                "Why I skipped a live crowd forecast",
                "Your wording did not ask how crowded a named ASI monument would be, "
                "so I did not treat this as a prediction-only question. "
                "I still check history when you name trip anchors or a region with no catalog match.",
            )

    if "search_destinations" in trace:
        add(
            "Why I searched local destination data",
            f"I searched our ASI + Kaggle catalogs for {dest} "
            f"{'and ' + ', '.join(must) if must else ''} so recommendations come from tools, not memory.",
        )

    if "web_search" in trace:
        queries = web.get("queries") or []
        add(
            "Why I used web search",
            "I searched because you gave trip dates, asked about current conditions/events, "
            "or local catalogs did not yield enough nearby places. "
            "Preferred official hosts (state tourism, Incredible India, ASI). "
            + (f"Queries: {'; '.join(queries[:4])}. " if queries else "")
            + "Snippets are current/contextual facts only — never visitor forecasts. "
            "Recruitment/PDF junk is dropped.",
        )
    elif intent.get("needs_current_context"):
        add(
            "Web search",
            "Current context was requested but no usable hits were kept after filters.",
        )
    else:
        add(
            "Why I did not lead with the web",
            "You did not ask for weather, events, or dated trip conditions, and "
            "I already had enough local data for the core question.",
        )

    pressure = state.get("period_pressure") or {}
    if pressure.get("found"):
        add(
            "Why I commented on your travel dates",
            f"Window {pressure.get('start_date')} to {pressure.get('end_date')}: "
            f"season={pressure.get('season')}, "
            f"weekends={pressure.get('weekend_days')}, "
            f"direction={pressure.get('footfall_direction')}. "
            "This is HEURISTIC/WEB FACT pressure on the annual ASI number — not a new daily headcount.",
        )

    if intent.get("needs_alternatives"):
        if ranked:
            add(
                "How I chose alternatives",
                "Geography is the first filter (same city/district/nearby km). "
                "Same ASI administrative circle is only a heritage-similarity hint, not proof two places are nearby. "
                "Each kept place must be relevant to your interests and practical for the trip. "
                "Crowd-backed = we have an ASI persistence forecast. "
                "Discovery = useful nearby place with no official visitor series.",
            )
            for r in ranked[:8]:
                mode = r.get("recommendation_mode") or "discovery"
                dist = r.get("distance_km")
                dist_s = f"{dist} km" if dist is not None else "distance unknown (not invented)"
                why = "; ".join((r.get("why") or [])[:4])
                add(
                    f"Why I recommend {r.get('name')}",
                    f"Mode={mode}. Practical={r.get('practical')}, relevant={r.get('relevant')}, "
                    f"geography_verified={r.get('geography_verified')}, geo_tier={r.get('geo_tier')}, "
                    f"{dist_s}. Crowd data available={r.get('crowd_data_available')}. "
                    f"Source={r.get('source')}. {why}",
                )
        else:
            add(
                "Why the alternative list is empty or short",
                intent.get("candidate_note")
                or "No candidate passed both relevance and verified proximity. "
                "I did not fill the list with far-away same-state sites just to reach a quota.",
            )

    if plan:
        add(
            "How I built the day plan",
            "Must-visit names stay as anchors. Remaining days are orbits from the ranked nearby list. "
            "This is a sketch of what to see, not an occupancy-optimised timetable.",
        )
        for day in plan:
            add(
                f"Why {day.get('date')} is {day.get('destination')}",
                f"Slot type={day.get('type')}. {day.get('reason')}",
            )

    add(
        "What I refused to do",
        "I will not invent hourly headcount, occupancy %, or ASI visitor totals for unmatched names. "
        "I will not treat a web adjective as a crowd metric. "
        "I will not call a distant city 'nearby' only because it shares an ASI circle.",
    )

    if extra:
        add("Planner notes (this run)", " | ".join(extra))

    return lines


def predict_was_checked(forecasts: dict, must: list, dest) -> bool:
    return bool(forecasts)
