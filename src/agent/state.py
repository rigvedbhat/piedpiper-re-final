"""LangGraph state for the tourist intelligence agent."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class IntentFlags(TypedDict, total=False):
    needs_prediction: bool
    needs_alternatives: bool
    needs_current_context: bool
    needs_time_recommendation: bool
    needs_trip_plan: bool
    wants_hourly: bool
    fallback_discovery: bool
    requested_n: int


class TourismState(TypedDict, total=False):
    user_request: str
    conversation: str
    awaiting_user: bool
    pending_question: str
    missing_slots: list[str]
    destination: str | None
    place_names: list[str]
    dest_state: str | None
    climate_zone: str | None
    web_queries: list[str]
    web_raw_hits: list[dict]
    brain_source: str
    brain_ran_tools: bool
    tool_events: list[dict]
    decisions: list[str]
    start_date: str | None
    end_date: str | None
    interests: list[str]
    must_visit: list[str]
    party_type: str | None
    wants_alternatives: bool
    wants_hourly: bool
    needs_web: bool
    is_multi_day: bool
    intent: IntentFlags

    data_granularity: dict
    destinations: list[dict]
    profiles: dict
    historical_demand: dict
    forecasts: dict
    crowd_levels: dict
    web_context: dict
    sources: list[dict]
    candidate_alternatives: list[dict]
    ranked_alternatives: list[dict]
    trip_plan: list[dict]
    tool_trace: list[str]
    critic_retries: int
    critic_done: bool
    period_pressure: dict
    brain_steps: int
    brain_response: str
    final_response: str
    answer_source: str
    messages: Annotated[list[Any], add_messages]
