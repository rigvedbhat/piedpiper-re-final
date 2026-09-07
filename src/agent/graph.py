"""LangGraph tourist agent: structured intent, then a Python tool plan."""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from src.agent.brain import _llm_enabled, decide_next_tools, run_llm_tool_loop
from src.agent.compose import compose_answer
from src.agent.critic import MAX_CRITIC_RETRIES, critic_has_gap, critic_patches
from src.agent.intent import resolve_intent
from src.agent.planner import execute_plan
from src.agent.state import TourismState

load_dotenv(Path(".env"), override=True)


def _text_from_messages(messages: list) -> str:
    parts: list[str] = []
    for msg in messages or []:
        if isinstance(msg, dict):
            role = (msg.get("type") or msg.get("role") or "").lower()
            content = msg.get("content") or ""
        else:
            role = (getattr(msg, "type", None) or "").lower()
            content = getattr(msg, "content", "") or ""
        if role in {"human", "user"} and str(content).strip():
            parts.append(str(content).strip())
    return parts[-1] if parts else ""


def coerce_user_request(state: dict) -> str:
    req = (state.get("user_request") or state.get("conversation") or "").strip()
    if req:
        return req
    return _text_from_messages(state.get("messages") or [])


def parse_intent(state: TourismState) -> dict:
    """Prepare empty research buckets; the live brain interprets the trip."""
    request = coerce_user_request(state)
    buckets = {
        "tool_trace": ["parse_intent"],
        "critic_retries": 0,
        "data_granularity": {"supported": "annual", "hourly": False, "daily": False},
        "sources": [],
        "forecasts": {},
        "crowd_levels": {},
        "profiles": {},
        "messages": [],
        "decisions": [],
        "destinations": [],
        "ranked_alternatives": [],
        "candidate_alternatives": [],
        "trip_plan": [],
        "web_context": {},
        "period_pressure": {},
        "user_request": request,
        "conversation": state.get("conversation") or request,
    }
    if _llm_enabled():
        return {
            **buckets,
            "destination": None,
            "start_date": None,
            "end_date": None,
            "interests": [],
            "must_visit": [],
            "place_names": [],
            "intent": {},
            "awaiting_user": False,
        }
    parsed = resolve_intent(request)
    return {**buckets, **parsed}


def brain_intent_node(state: TourismState) -> dict:
    """OpenAI chooses every tool and argument. Offline: flags only, then Python execute_plan."""
    if _llm_enabled():
        return run_llm_tool_loop(state)
    choice = decide_next_tools(state)
    intent = dict(state.get("intent") or {})
    tools = set(choice.get("tools") or [])
    request = (state.get("user_request") or "").strip().lower()
    if "forecast_crowd" in tools and not request.startswith("who is"):
        intent["needs_prediction"] = True
    if state.get("start_date") or intent.get("needs_current_context"):
        if "web_search" in tools or "assess_period_pressure" in tools or state.get("start_date"):
            intent["needs_current_context"] = True
    if "rank_alternatives" in tools or "find_similar_destinations" in tools:
        intent["needs_alternatives"] = True
    if "optimize_trip" in tools:
        intent["needs_trip_plan"] = True
        intent["needs_alternatives"] = True
        intent["needs_current_context"] = True
    return {
        "intent": intent,
        "brain_steps": int(state.get("brain_steps") or 0) + 1,
        "brain_source": choice.get("source"),
        "brain_ran_tools": False,
        "tool_trace": list(state.get("tool_trace") or []) + ["llm_brain"],
    }


def execute_plan_node(state: TourismState) -> dict:
    if (state.get("brain_source") != "fallback" and _llm_enabled()) or state.get("brain_ran_tools"):
        return {}
    return execute_plan(state)


def route_after_brain(state: TourismState) -> str:
    if state.get("awaiting_user"):
        return "compose_response"
    if state.get("brain_source") == "fallback":
        return "execute_plan"
    if _llm_enabled() or state.get("brain_ran_tools"):
        return "compose_response"
    return "execute_plan"


def compose_node(state: TourismState) -> dict:
    from src.agent.compose import compose_answer_with_source

    response, source = compose_answer_with_source(state)
    return {
        "final_response": response,
        "answer_source": source,
        "tool_trace": list(state.get("tool_trace") or []) + ["compose_response"],
    }


def critic_node(state: TourismState) -> dict:
    patches = critic_patches(state)
    intent = dict(state.get("intent") or {})
    dest = state.get("destination")
    out: dict = {
        "critic_retries": int(state.get("critic_retries") or 0) + 1,
        "tool_trace": list(state.get("tool_trace") or []) + ["critic_retry"],
    }
    if patches.get("_destination"):
        dest = patches["_destination"]
        out["destination"] = dest
    if patches.get("_interests"):
        out["interests"] = patches["_interests"]
    if patches.get("_must_visit"):
        out["must_visit"] = patches["_must_visit"]
    if patches.get("_start_date"):
        out["start_date"] = patches["_start_date"]
    if patches.get("_end_date"):
        out["end_date"] = patches["_end_date"]
    for key, val in patches.items():
        if key.startswith("_"):
            continue
        intent[key] = val
    out["intent"] = intent
    return out


def after_plan(state: TourismState) -> str:
    if _llm_enabled() or state.get("brain_ran_tools"):
        return "compose_response"
    retries = int(state.get("critic_retries") or 0)
    if retries >= MAX_CRITIC_RETRIES:
        return "compose_response"
    if critic_has_gap(state):
        return "critic"
    return "compose_response"


def ingest_tool_results(state: TourismState) -> dict:
    """Copy structured tool JSON into state buckets. Kept for unit tests."""
    forecasts = dict(state.get("forecasts") or {})
    crowds = dict(state.get("crowd_levels") or {})
    profiles = dict(state.get("profiles") or {})
    sources = list(state.get("sources") or [])
    ranked = list(state.get("ranked_alternatives") or [])
    candidates = list(state.get("candidate_alternatives") or [])
    historical = dict(state.get("historical_demand") or {})
    web_context = dict(state.get("web_context") or {})
    trace = list(state.get("tool_trace") or [])

    for msg in state.get("messages") or []:
        if not isinstance(msg, ToolMessage):
            continue
        name = getattr(msg, "name", "") or ""
        try:
            payload = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            continue
        if name.startswith("forecast_crowd"):
            dest = payload.get("destination")
            if dest:
                forecasts[dest] = payload
        elif name.startswith("classify_crowd"):
            dest = payload.get("destination")
            if dest:
                crowds[dest] = payload
        elif name.startswith("get_destination_profile"):
            key = payload.get("asi_name") or payload.get("query")
            if key:
                profiles[key] = payload
        elif name.startswith("get_historical_footfall"):
            dest = payload.get("destination") or payload.get("query")
            if dest:
                historical[dest] = payload
        elif name.startswith("rank_alternatives"):
            if isinstance(payload, list):
                ranked = payload
        elif name.startswith("find_similar"):
            if isinstance(payload, list):
                candidates = payload
        elif name.startswith("web_search"):
            web_context = payload
            for hit in payload.get("results") or []:
                if hit.get("url"):
                    sources.append(
                        {
                            "title": hit.get("title"),
                            "url": hit.get("url"),
                            "domain": hit.get("source"),
                            "published": hit.get("published"),
                            "snippet": hit.get("snippet"),
                        }
                    )
        if name:
            short = name.replace("_tool", "")
            if not trace or trace[-1] != short:
                trace.append(short)
    return {
        "forecasts": forecasts,
        "crowd_levels": crowds,
        "profiles": profiles,
        "sources": sources,
        "ranked_alternatives": ranked,
        "candidate_alternatives": candidates,
        "historical_demand": historical,
        "web_context": web_context,
        "tool_trace": trace,
    }


def route_after_agent(state: TourismState) -> str:
    """Legacy ReAct helper retained for tests."""
    last = (state.get("messages") or [None])[-1]
    agent_rounds = sum(1 for t in state.get("tool_trace") or [] if t == "agent")
    keep_tools = (
        isinstance(last, AIMessage)
        and bool(getattr(last, "tool_calls", None))
        and agent_rounds < 4
    )
    if keep_tools:
        return "tools"
    if state.get("is_multi_day"):
        return "optimize_trip"
    return "finalize"


def build_graph():
    g = StateGraph(TourismState)
    g.add_node("parse_intent", parse_intent)
    g.add_node("llm_brain", brain_intent_node)
    g.add_node("execute_plan", execute_plan_node)
    g.add_node("critic", critic_node)
    g.add_node("compose_response", compose_node)
    g.add_edge(START, "parse_intent")
    g.add_edge("parse_intent", "llm_brain")
    g.add_conditional_edges(
        "llm_brain",
        route_after_brain,
        {"compose_response": "compose_response", "execute_plan": "execute_plan"},
    )
    g.add_conditional_edges(
        "execute_plan",
        after_plan,
        {"critic": "critic", "compose_response": "compose_response"},
    )
    g.add_edge("critic", "execute_plan")
    g.add_edge("compose_response", END)
    return g.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_agent(
    user_request: str,
    recursion_limit: int = 32,
    conversation: str | None = None,
) -> TourismState:
    import uuid

    graph = get_graph()
    # Fresh thread so Chat UI's messages checkpoint is not reused (that dropped user_request).
    result = graph.invoke(
        {
            "user_request": user_request,
            "conversation": conversation or user_request,
            "tool_trace": [],
        },
        {
            "recursion_limit": recursion_limit,
            "configurable": {"thread_id": f"tourism-inner-{uuid.uuid4()}"},
        },
    )
    return result


def format_cli(state: TourismState, verbose: bool = False) -> str:
    intent = state.get("intent") or {}
    lines = [
        "========================================================",
        "          TOURISM CROWD INTELLIGENCE AGENT",
        "========================================================",
        "",
        "TRIP REQUEST",
        f"Destination: {state.get('destination') or '(from request)'}",
        f"Dates: {state.get('start_date') or '-'} -> {state.get('end_date') or '-'}",
        f"Interests: {', '.join(state.get('interests') or []) or '-'}",
        f"Must-visit: {', '.join(state.get('must_visit') or []) or '-'}",
        "",
    ]
    if verbose:
        lines += [
            "USER INTENT",
            f"prediction: {bool(intent.get('needs_prediction'))}",
            f"alternatives: {bool(intent.get('needs_alternatives'))}",
            f"current_context: {bool(intent.get('needs_current_context'))}",
            f"trip_plan: {bool(intent.get('needs_trip_plan'))}",
            f"fallback_discovery: {bool(intent.get('fallback_discovery'))}",
            "",
            "TOOL TRACE",
            " -> ".join(state.get("tool_trace") or []),
            "",
        ]
    lines += [
        "--------------------------------------------------------",
        "ANSWER",
        "--------------------------------------------------------",
        "",
        state.get("final_response") or "(no response)",
        "",
    ]
    if state.get("sources"):
        lines += ["--------------------------------------------------------", "SOURCES", ""]
        for s in state["sources"][:8]:
            lines.append(f"- {s.get('title')} | {s.get('url')}")
        lines.append("")
    lines.append("========================================================")
    return "\n".join(lines)
