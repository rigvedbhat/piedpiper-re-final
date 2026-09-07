"""Traveler-facing answer. Live path: Gemini brain writes conversational response.
Offline fallback: template-based structured answer."""

from __future__ import annotations

import json
import os

from src.agent.reasoning import build_decision_reasoning
from src.agent.research import nearby_from_research
from src.agent.state import TourismState

ANSWER_SYSTEM = """You are chatting with a traveler as their helpful travel buddy. Write naturally, like a knowledgeable friend — not a report or a form.

RESEARCH_FACTS is the only evidence you can use.

RULES:
1. If awaiting_user is true and there are no forecasts:
   - Just continue the conversation. Ask for what's missing naturally.
   - Do NOT dump a crowd report or invent any place/dates/party info.

2. If awaiting_user is true AND forecasts or ranked places exist:
   - Share those findings conversationally, then ask for the missing info naturally.

3. If research has run:
   - Acknowledge what they told you (place, dates, travel companions).
   - Share your crowd prediction naturally with reasoning (mention season, events, holidays, weekday/weekend patterns).
   - If you know specific peak days or calmer windows, mention them.
   - Suggest best visiting hours if the data supports it (mornings are generally less crowded for popular monuments).
   - If alternatives exist, recommend them naturally:
     • Name each alternative with a brief "why" (1-2 sentences)
     • Clearly label any hidden gems or underrated spots
     • Mention approximate distance from the main destination
   - Keep it concise and conversational. 200-400 words max.
   - Don't use section headers like "CONCLUSION" or "CROWD FORECAST".
   - Copy exact visitor numbers from facts if present. Never invent numbers.
   - If no ASI data exists, say so honestly but still reason from web features.
   - End with a friendly note or a natural follow-up question.
"""


def _fmt_visitors(value) -> str:
    if value is None:
        return "not available"
    try:
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return str(value)


def _brief_conclusion(state: TourismState) -> list[str]:
    dest = state.get("destination") or "this trip"
    party = state.get("party_type")
    start, end = state.get("start_date"), state.get("end_date")
    forecasts = state.get("forecasts") or {}
    crowds = state.get("crowd_levels") or {}
    historical = state.get("historical_demand") or {}
    pressure = state.get("period_pressure") or {}
    ranked = state.get("ranked_alternatives") or []
    plan = state.get("trip_plan") or []
    found = [v for v in forecasts.values() if v.get("found")]
    lines = ["CONCLUSION"]
    who = f" ({party})" if party else ""
    if start:
        lines.append(
            f"You asked about {dest}{who} from {start} to {end or start}."
        )
    else:
        lines.append(f"You asked about {dest}{who}.")

    if found:
        fc = found[0]
        name = fc.get("destination") or dest
        crowd = crowds.get(name) or crowds.get(fc.get("destination") or "") or {}
        hist = historical.get(name) or historical.get(dest) or {}
        direction = hist.get("annual_direction") or fc.get("trend") or "unknown"
        lines.append(
            f"According to our analysis of official ASI annual history, {name} "
            f"has persistence demand {_fmt_visitors(fc.get('predicted_visitors'))} "
            f"visitors ({fc.get('forecast_period')}). Relative level: "
            f"{crowd.get('level') or 'n/a'} (not occupancy %)."
        )
        if direction and direction != "unknown":
            lines.append(
                f"The recent annual series looks {direction} — that is year-to-year ticketed "
                "footfall, not a daily prediction."
            )
        if pressure.get("found"):
            lines.append(
                f"For this travel window, pressure vs that annual baseline looks "
                f"{pressure.get('footfall_direction')} "
                f"(season={pressure.get('season')}; "
                f"{pressure.get('weekend_days')} weekend / {pressure.get('weekday_days')} weekday). "
                "This does not replace the ASI number."
            )
        research = nearby_from_research(state)
        if research.get("suggest_nearby") and ranked:
            lines.append(
                "Because the research step (window, weather/events if found, history if any) "
                "points to extra pressure, nearby/underrated places are listed below: "
                + "; ".join(research.get("reasons") or [])
                + "."
            )
        elif plan:
            lines.append("A sketch schedule from your dates is below.")
    else:
        lines.append(
            "According to our analysis, this name is not in the ASI monument visitor tables, "
            "so we do not invent a visitor total."
        )
        if pressure.get("found"):
            lines.append(
                f"Travel-window research (season/weekends/events) suggests "
                f"{pressure.get('footfall_direction')} pressure vs a typical year — heuristic, not a count."
            )
        if ranked:
            lines.append(
                "Nearby places below are research recommendations (catalog + web), not occupancy ranks."
            )
        if plan:
            lines.append("You can follow this sketch schedule; keep days flexible where we lack nearby orbits.")
    lines.append("")
    return lines


def compose_final_response(state: TourismState) -> str:
    """Conversational fallback when LLM API is unavailable."""
    dest = state.get("destination") or "your destination"
    party = f" with your {state['party_type']}" if state.get("party_type") else ""
    dates = f" from {state['start_date']} to {state.get('end_date') or state['start_date']}" if state.get("start_date") else ""
    forecasts = state.get("forecasts") or {}
    crowds = state.get("crowd_levels") or {}
    pressure = state.get("period_pressure") or {}
    ranked = state.get("ranked_alternatives") or []

    parts = [f"I've analyzed the travel and crowd conditions for **{dest}**{party}{dates}."]
    found = [v for v in forecasts.values() if v.get("found")]
    if found:
        fc = found[0]
        c = crowds.get(fc.get("destination") or dest) or {}
        parts.append(
            f"**Crowd Forecast**: Expected annual demand is approximately {fc.get('predicted_visitors'):,.0f} visitors ({fc.get('forecast_period')}), with a relative crowd density of **{c.get('level', 'MODERATE')}**."
        )
    else:
        parts.append(
            f"**Crowd Forecast**: {dest} is outside the official ASI ticketed monument catalog, so historical visitor counts are uncataloged. Based on typical seasonal patterns, expect steady tourist flow."
        )
    if pressure.get("found"):
        parts.append(
            f"**Travel Window**: Footfall pressure is **{pressure.get('footfall_direction', 'normal')}** ({pressure.get('weekend_days', 0)} weekend day(s), season: {pressure.get('season')}). Weekends and midday hours (11 AM – 4 PM) will be the most crowded."
        )
        parts.append("**Best Visiting Hours**: Early morning (6:30 AM – 8:30 AM) or after 4:30 PM for the most peaceful experience.")
    if ranked:
        recs = []
        for r in ranked[:4]:
            tag = "Hidden Gem" if r.get("recommendation_mode") == "discovery" else "Alternative"
            dist = f", ~{r.get('distance_km')} km" if r.get("distance_km") else ""
            recs.append(f"- **{r.get('name')}** ({tag}{dist})")
        parts.append("**Nearby Places to Consider**:\n" + "\n".join(recs))
    parts.append("Let me know if you'd like more specific recommendations or timing details!")
    return "\n\n".join(parts)


def _answer_facts(state: TourismState) -> dict:
    web = state.get("web_context") or {}

    def _hits(key: str) -> list[dict]:
        out = []
        for hit in (web.get(key) or [])[:4]:
            out.append(
                {
                    "title": hit.get("title"),
                    "url": hit.get("url"),
                    "snippet": (hit.get("snippet") or "")[:180],
                    "evidence_tier": hit.get("evidence_tier"),
                }
            )
        return out

    ranked = []
    for r in (state.get("ranked_alternatives") or [])[:8]:
        ranked.append(
            {
                "name": r.get("name"),
                "distance_km": r.get("distance_km"),
                "why": (r.get("why") or [])[:4],
                "recommendation_mode": r.get("recommendation_mode"),
                "predicted_visitors": r.get("predicted_visitors"),
                "crowd_data_available": r.get("crowd_data_available"),
            }
        )
    hist = {}
    for k, v in (state.get("historical_demand") or {}).items():
        hist[k] = {
            "found": (v or {}).get("found"),
            "annual_direction": (v or {}).get("annual_direction"),
            "n_years": len((v or {}).get("observations") or []),
        }
    forecasts = {}
    for k, v in (state.get("forecasts") or {}).items():
        forecasts[k] = {
            "found": (v or {}).get("found"),
            "destination": (v or {}).get("destination"),
            "predicted_visitors": (v or {}).get("predicted_visitors"),
            "forecast_period": (v or {}).get("forecast_period"),
            "forecast_method": (v or {}).get("forecast_method"),
            "trend": (v or {}).get("trend"),
            "layer": (v or {}).get("layer"),
        }
    return {
        "user_request": (state.get("user_request") or "")[:2000],
        "conversation": (state.get("conversation") or "")[:4000],
        "awaiting_user": bool(state.get("awaiting_user")),
        "pending_question": state.get("pending_question"),
        "missing_slots": state.get("missing_slots") or [],
        "destination": state.get("destination"),
        "place_names": state.get("place_names"),
        "dest_state": state.get("dest_state"),
        "party_type": state.get("party_type"),
        "start_date": state.get("start_date"),
        "end_date": state.get("end_date"),
        "must_visit": state.get("must_visit"),
        "interests": state.get("interests"),
        "forecasts": forecasts,
        "crowd_levels": {
            k: {"level": (v or {}).get("level"), "found": (v or {}).get("found")}
            for k, v in (state.get("crowd_levels") or {}).items()
        },
        "historical": hist,
        "period_pressure": {
            "found": (state.get("period_pressure") or {}).get("found"),
            "season": (state.get("period_pressure") or {}).get("season"),
            "weekend_days": (state.get("period_pressure") or {}).get("weekend_days"),
            "weekday_days": (state.get("period_pressure") or {}).get("weekday_days"),
            "holiday_dates": (state.get("period_pressure") or {}).get("holiday_dates"),
            "footfall_direction": (state.get("period_pressure") or {}).get("footfall_direction"),
            "reasons": (state.get("period_pressure") or {}).get("reasons"),
        },
        "web": {
            "weather": _hits("weather"),
            "events": _hits("events"),
            "holidays": _hits("holidays"),
            "closures": _hits("closures"),
            "advisories": _hits("advisories"),
        },
        "nearby_from_research": nearby_from_research(state),
        "ranked_alternatives": ranked,
        "trip_plan": state.get("trip_plan") or [],
    }


def llm_write_answer(state: TourismState) -> str | None:
    """Fallback answer composition via Gemini when the brain didn't write a response."""
    if os.environ.get("TOURISM_LLM_BRAIN", "1") != "1":
        return None
    if not (os.getenv("GOOGLE_API_KEY") or "").strip():
        return None
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception:
        return None
    model = os.getenv("GOOGLE_MODEL", "gemini-3.6-flash")
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.45,
        max_output_tokens=900,
    )
    try:
        msg = llm.invoke(
            [
                SystemMessage(content=ANSWER_SYSTEM),
                HumanMessage(content=json.dumps(_answer_facts(state), default=str)[:14000]),
            ]
        )
        def _extract(content):
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        if item.get("type") == "text" and item.get("text"):
                            parts.append(item["text"])
                        elif "text" in item:
                            parts.append(str(item["text"]))
                return "\n".join(parts).strip()
            return str(content or "").strip()

        text = _extract(getattr(msg, "content", None))
        return text or None
    except Exception:
        return None


def compose_answer_with_source(state: TourismState) -> tuple[str, str]:
    """Return the response plus the component that authored it."""
    # Priority 1: Brain already wrote a conversational response during tool loop
    brain_response = (state.get("brain_response") or "").strip()
    if brain_response:
        return brain_response, "llm_brain"

    # Priority 2: Awaiting user — return the pending question
    if state.get("awaiting_user") and (state.get("pending_question") or "").strip():
        return str(state["pending_question"]).strip(), "llm_brain_clarification"

    # Priority 3: Fallback LLM answer composition (separate call, only if brain didn't respond)
    live = llm_write_answer(state)
    if live:
        return live, "llm_compose"

    # Priority 4: Deterministic template
    return compose_final_response(state), "offline_template"


def compose_answer(state: TourismState) -> str:
    """Compatibility wrapper for callers that only need response text."""
    return compose_answer_with_source(state)[0]


def lookup_has_coords_note(ranked: list) -> bool:
    return all(r.get("distance_km") is not None for r in ranked) if ranked else True


def _append_web(lines: list[str], web: dict) -> None:
    if not web:
        lines.append("No current-context search was required or no usable hits remained after filtering.")
        return
    lines.append(f"Provider: {web.get('provider')} | retrieved_on: {web.get('retrieved_on')}")
    if web.get("queries"):
        lines.append("Queries: " + "; ".join(web.get("queries")[:4]))
    any_bucket = False
    for key in ("weather", "events", "holidays", "closures", "timings", "advisories"):
        hits = web.get(key) or []
        if not hits:
            continue
        any_bucket = True
        lines.append(f"{key}:")
        for hit in hits[:3]:
            tier = hit.get("evidence_tier") or "general_web"
            snippet = (hit.get("snippet") or "")[:160]
            lines.append(f"- [{tier}] {hit.get('title')} | {hit.get('url')}")
            if snippet:
                lines.append(f"  {snippet}")
    leftover = web.get("discovery") or []
    official_disc = [h for h in leftover if h.get("evidence_tier") == "official"]
    if official_disc and not any_bucket:
        lines.append("official destination pages (not classified as weather/events):")
        for hit in official_disc[:4]:
            lines.append(f"- [official] {hit.get('title')} | {hit.get('url')}")
    elif not any_bucket and leftover:
        lines.append("No weather/events/closures matched the trip dates. Unclassified hits were not used as event facts.")
    for w in web.get("warnings") or []:
        lines.append(f"Note: {w}")
    if not (any_bucket or official_disc or leftover or web.get("warnings")):
        lines.append("No current-context facts retained after junk/relevance filters.")
    lines.append("Web facts are not MODEL OUTPUT and are not crowd counts.")
