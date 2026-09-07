"""Autonomous LLM Brain for Tourist Crowd Prediction System.
The LLM dynamically receives traveler input, extracts intent, selects tools,
gathers dataset and web features, and synthesizes conversational recommendations.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from src.agent.dispatch import run_named_tool

logger = logging.getLogger(__name__)

LLM_DIRECTOR_PROMPT = """You are the autonomous Brain of the Tourist Crowd Prediction System.
The traveler has sent a message in an ongoing conversation.

Your job:
1. UNDERSTAND THE TRAVELER'S INTENT:
   - Identify destination (city, monument, town, or state).
   - Identify exact travel dates / duration (e.g. "5th of Sept. to 10th of Sept. 2026" -> start_date: "2026-09-05", end_date: "2026-09-10"). Use YYYY-MM-DD format. Do NOT make mistakes on dates.
   - Identify party type: "family", "friends", "solo", "couple", or "not specified".
   - Identify traveler interests (e.g. temples, beaches, heritage, hidden gems, nature).

2. DECIDE IF CLARIFICATION IS NEEDED:
   - If the user has NOT named a destination yet (or is just greeting e.g. "hi", "help me plan a trip"), set "is_clarification_needed": true and write a warm, friendly "clarification_message" welcoming them and asking where they are heading, their dates, and who they are traveling with.
   - If a destination IS mentioned, set "is_clarification_needed": false.

3. DYNAMICALLY SELECT TOOLS TO CALL:
   If a destination is identified, choose the tools to research and predict crowd density:
   - "get_destination_profile": {{"destination": "<destination>"}} (checks if destination is in ASI/Kaggle catalogs)
   - "forecast_crowd": {{"destination": "<destination>"}} (checks official ASI persistence forecast)
   - "web_search": {{"query": "<destination> weather festivals events holidays <month> <year>"}} (searches for live context, festivals, weather, holidays)
   - "assess_period_pressure": {{"destination": "<destination>", "start_date": "<start_date>", "end_date": "<end_date>"}} (analyzes weekends, weekdays, and seasonal pressure)
   - "find_similar_destinations": {{"destination": "<destination>", "interests": "<interests>"}} (finds candidate alternatives in same orbit)
   - "rank_alternatives": {{"anchor": "<destination>", "interests": "<interests>"}} (ranks mainstream and hidden gem alternatives)

RETURN ONLY A VALID JSON OBJECT (no markdown, no backticks):
{{
  "reasoning": "1-2 sentence explanation of intent and tool selection",
  "trip_context": {{
    "destination": "...",
    "start_date": "YYYY-MM-DD or null",
    "end_date": "YYYY-MM-DD or null",
    "party_type": "family|friends|solo|couple|not specified",
    "interests": ["..."]
  }},
  "is_clarification_needed": false,
  "clarification_message": "",
  "tool_calls": [
    {{"tool": "get_destination_profile", "args": {{"destination": "..."}}}},
    {{"tool": "forecast_crowd", "args": {{"destination": "..."}}}},
    {{"tool": "web_search", "args": {{"query": "..."}}}},
    {{"tool": "assess_period_pressure", "args": {{"destination": "...", "start_date": "...", "end_date": "..."}}}},
    {{"tool": "find_similar_destinations", "args": {{"destination": "..."}}}},
    {{"tool": "rank_alternatives", "args": {{"anchor": "..."}}}}
  ]
}}

Conversation History:
{conversation}

Latest Traveler Message:
"{user_request}"
"""

BRAIN_CONVERSATIONAL_PROMPT = """You are a knowledgeable, friendly, and expert travel companion and crowd prediction assistant.
You talk naturally with the traveler — warm, professional, engaging, and conversational.
NEVER sound like a robotic template, and NEVER output section titles like "CONCLUSION", "CROWD FORECAST", "LIMITATIONS", or "HOW I REASONED". Just talk naturally using clean markdown formatting (bullet points, bold text).

HERE IS THE RESEARCH DATA GATHERED FOR THIS TRIP:
{research_summary}

YOUR TASK:
1. Warmly acknowledge their trip: Destination, travel window (if given), and traveling style (family, friends, solo).
2. Crowd Prediction & Reasoning:
   - State whether the crowd density during their visit duration will be HIGH, MODERATE, or LOW.
   - Ground your reasoning in the actual features:
     • Official ASI monument persistence demand and historical trend (if in the dataset)
     • Weather conditions during that period
     • Any festivals, events, or local celebrations happening
     • Gazetted / school holidays
     • Weekday vs weekend distribution
   - If the place is NOT in the official ASI dataset, clearly mention that official ticketed history is uncataloged, but give your crowd prediction based on web features, seasonal trends, and weekends/holidays.
3. Specific Peak Days & Best Calm Hours:
   - Identify which specific days in their window are likely to see the heaviest rush (e.g., Saturday/Sunday, festival days).
   - Recommend the best visiting hours (e.g., early morning 6:00 AM – 8:30 AM or late afternoon) when they won't face heavy crowds.
4. Nearby Alternative & Underrated Destinations:
   - Recommend 3 to 5 nearby places to explore.
   - Clearly label which ones are mainstream alternatives and which ones are **underrated / hidden gems**.
   - Mention approximate distance from the main destination and give a 1–2 sentence reason why each is worth visiting.
5. Friendly Closing:
   - End with a welcoming note inviting any questions or itinerary refinements.

Keep your response clean, engaging, informative, and around 250–400 words.
"""


def _llm_enabled() -> bool:
    if os.environ.get("TOURISM_LLM_BRAIN", "1") != "1":
        return False
    has_google = bool((os.getenv("GOOGLE_API_KEY") or "").strip())
    has_openrouter = bool((os.getenv("OPENROUTER_API_KEY") or "").strip())
    return has_google or has_openrouter


def _extract_text(msg: Any) -> str:
    content = getattr(msg, "content", msg)
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


def _parse_llm_json(raw: str) -> dict | None:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return None


def _invoke_llm(prompt: str, temperature: float = 0.2, max_tokens: int = 4000) -> str:
    """Invoke LLM with resilient model fallback (Google Gemini -> OpenRouter)."""
    google_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    models_to_try = [
        os.getenv("GOOGLE_MODEL", "gemini-3-flash-preview"),
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
        "gemini-flash-latest",
    ]

    # Try Google Gemini first
    if google_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        seen_models = set()
        for m in models_to_try:
            if not m or m in seen_models:
                continue
            seen_models.add(m)
            try:
                llm = ChatGoogleGenerativeAI(
                    model=m,
                    google_api_key=google_key,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                msg = llm.invoke(prompt)
                text = _extract_text(msg)
                if text:
                    return text
            except Exception as exc:
                logger.warning(f"Gemini model {m} failed: {exc}")
                continue

    # Fallback to OpenRouter if configured
    openrouter_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if openrouter_key:
        try:
            from langchain_openai import ChatOpenAI

            model = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
            llm = ChatOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
                model=model,
                temperature=temperature,
                max_tokens=min(max_tokens, 1500),
            )
            msg = llm.invoke(prompt)
            text = _extract_text(msg)
            if text:
                return text
        except Exception as exc:
            logger.warning(f"OpenRouter fallback failed: {exc}")

    raise RuntimeError("All LLM providers and models failed.")


def run_llm_tool_loop(state: dict) -> dict:
    """Autonomous LLM Brain tool calling loop.
    Turn 1: LLM interprets intent & chooses tools.
    Python executes chosen tools and gathers observations.
    Turn 2: LLM synthesizes findings into conversational travel recommendation.
    """
    working = dict(state)
    working.setdefault("decisions", [])
    working.setdefault("tool_trace", [])
    working.setdefault("tool_events", [])

    user_req = (working.get("user_request") or "").strip()
    conversation = (working.get("conversation") or user_req).strip()

    if not _llm_enabled():
        from src.agent.compose import compose_final_response
        resp = compose_final_response(working)
        working["final_response"] = resp
        working["brain_response"] = resp
        working["answer_source"] = "offline_template"
        working["brain_source"] = "offline"
        return working

    # =========================================================================
    # Turn 1: LLM Brain interprets user intent and chooses tools to call
    # =========================================================================
    director_prompt = LLM_DIRECTOR_PROMPT.format(
        conversation=conversation[:4000],
        user_request=user_req,
    )
    try:
        director_raw = _invoke_llm(director_prompt, temperature=0.1, max_tokens=4000)
        plan = _parse_llm_json(director_raw) or {}
    except Exception as exc:
        working["decisions"].append(f"Director LLM error: {exc}")
        plan = {}

    trip_ctx = plan.get("trip_context") or {}
    dest = (trip_ctx.get("destination") or "").strip()
    start = (trip_ctx.get("start_date") or "").strip()
    end = (trip_ctx.get("end_date") or "").strip()
    party = (trip_ctx.get("party_type") or "not specified").strip()
    interests = trip_ctx.get("interests") or []

    # Update working state with brain-extracted context
    if dest:
        working["destination"] = dest
    if start:
        working["start_date"] = start
    if end:
        working["end_date"] = end
    if party:
        working["party_type"] = party
    if interests:
        working["interests"] = interests

    # Check if the traveler needs clarification
    if plan.get("is_clarification_needed") or not dest:
        clarification = (plan.get("clarification_message") or "").strip()
        if not clarification:
            clarification = (
                "Hello! I'd love to help you plan your trip. Which destination do you have in mind? "
                "Also, please let me know your planned travel dates and whether you're traveling solo, with family, or with friends!"
            )
        working["brain_response"] = clarification
        working["pending_question"] = clarification
        working["final_response"] = clarification
        working["awaiting_user"] = True
        working["answer_source"] = "llm_brain_clarification"
        working["brain_source"] = "llm"
        return working

    # =========================================================================
    # Execute LLM Brain's selected tools
    # =========================================================================
    tool_calls = plan.get("tool_calls") or []
    tool_events: list[dict] = list(working.get("tool_events") or [])
    observations: dict[str, Any] = {}

    # If the LLM didn't emit tool calls for a valid destination, provide essential ones
    if not tool_calls and dest:
        tool_calls = [
            {"tool": "get_destination_profile", "args": {"destination": dest}},
            {"tool": "forecast_crowd", "args": {"destination": dest}},
            {"tool": "web_search", "args": {"query": f"{dest} tourism weather events festivals holidays"}},
            {"tool": "assess_period_pressure", "args": {"destination": dest, "start_date": start, "end_date": end}},
            {"tool": "find_similar_destinations", "args": {"destination": dest, "interests": ", ".join(interests)}},
            {"tool": "rank_alternatives", "args": {"anchor": dest, "interests": ", ".join(interests)}},
        ]

    for item in tool_calls:
        tool_name = (item.get("tool") or "").strip()
        args = item.get("args") or {}
        if not tool_name:
            continue
        try:
            obs, patch = run_named_tool(tool_name, args, working)
            working.update(patch)
            observations[tool_name] = obs
            tool_events.append({"name": tool_name, "args": args, "observation": obs})
            working["tool_trace"].append(tool_name)
        except Exception as exc:
            observations[tool_name] = {"error": str(exc)}
            logger.warning(f"Error running tool {tool_name}: {exc}")

    working["tool_events"] = tool_events

    # =========================================================================
    # Compile research facts for the LLM
    # =========================================================================
    summary_lines = [
        f"- Destination: {dest}",
        f"- Travel Window: {start or 'Not specified'} to {end or start or 'Not specified'}",
        f"- Party Type: {party}",
        f"- Interests: {', '.join(interests) if interests else 'General sightseeing'}",
    ]

    # Profile & ASI forecast
    profile = observations.get("get_destination_profile") or working.get("profile") or {}
    has_asi = bool(profile.get("has_asi_history"))
    fc = observations.get("forecast_crowd") or {}
    if isinstance(fc, dict) and fc.get("found"):
        summary_lines.append(
            f"- Official ASI Annual Persistence Forecast: {fc.get('predicted_visitors'):,.0f} visitors ({fc.get('forecast_period')})"
        )
        cl = observations.get("classify_crowd") or {}
        if isinstance(cl, dict) and cl.get("level"):
            summary_lines.append(f"- Relative Crowd Level: {cl.get('level')} (historical quartile)")
        if fc.get("trend"):
            summary_lines.append(f"- Historical Trend: {fc.get('trend')}")
    else:
        summary_lines.append(f"- Official ASI Catalog: {dest} is not in the ASI annual ticketed monument registry (ticketing uncataloged).")

    # Kaggle metadata if available
    kaggle_meta = profile.get("kaggle") or {}
    if kaggle_meta:
        summary_lines.append(
            f"- Catalog Metadata: State={kaggle_meta.get('state')}, Best Season={kaggle_meta.get('season')}, Rating={kaggle_meta.get('google_rating')}"
        )

    # Web search results
    web_res = observations.get("web_search") or working.get("web_context") or {}
    if isinstance(web_res, dict):
        results = web_res.get("results") or []
        if results:
            summary_lines.append("- Live Web Context:")
            for r in results[:4]:
                t = r.get("title") or ""
                s = (r.get("snippet") or "")[:200]
                summary_lines.append(f"  • {t}: {s}")

    # Period pressure
    pressure = observations.get("assess_period_pressure") or working.get("period_pressure") or {}
    if isinstance(pressure, dict) and pressure.get("found"):
        summary_lines.append(
            f"- Travel Window Pressure: Footfall trend is {pressure.get('footfall_direction')} "
            f"({pressure.get('weekend_days')} weekend day(s), {pressure.get('weekday_days')} weekday(s), season={pressure.get('season')})"
        )
        if pressure.get("holiday_dates"):
            summary_lines.append(f"- Holidays in window: {', '.join(pressure.get('holiday_dates'))}")

    # Alternatives & Hidden Gems
    ranked = observations.get("rank_alternatives") or working.get("ranked_alternatives") or []
    if isinstance(ranked, list) and ranked:
        summary_lines.append("- Nearby Alternatives & Hidden Gems in Orbit:")
        for r in ranked[:5]:
            mode = "Hidden Gem" if r.get("recommendation_mode") == "discovery" else "Mainstream Alternative"
            dist = f"~{r.get('distance_km')} km" if r.get("distance_km") else "nearby"
            why = "; ".join(r.get("why") or [])
            summary_lines.append(f"  • {r.get('name')} ({mode}, {dist}): {why}")

    research_summary = "\n".join(summary_lines)

    # =========================================================================
    # Turn 2: LLM Brain synthesizes final conversational travel advisor response
    # =========================================================================
    synthesis_prompt = BRAIN_CONVERSATIONAL_PROMPT.format(research_summary=research_summary)
    try:
        final_answer = _invoke_llm(synthesis_prompt, temperature=0.3, max_tokens=2500)
        if final_answer:
            working["brain_response"] = final_answer
            working["final_response"] = final_answer
            working["answer_source"] = "llm_brain"
            working["brain_source"] = "llm"
            working["tool_trace"].append("llm_brain_synthesis")
            return working
    except Exception as exc:
        working["decisions"].append(f"Synthesis LLM error: {exc}")

    # Fallback if synthesis fails
    from src.agent.compose import compose_final_response
    fallback_resp = compose_final_response(working)
    working["brain_response"] = fallback_resp
    working["final_response"] = fallback_resp
    working["answer_source"] = "offline_template"
    working["brain_source"] = "fallback"
    return working


def decide_next_tools(state: dict) -> dict:
    return {"tools": [], "done": True, "source": "llm_brain"}
