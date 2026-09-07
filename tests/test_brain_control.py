"""OpenAI brain owns tool args; Python only executes."""

from __future__ import annotations

import unittest

from src.agent.brain import canonical_tool_name, fallback_tool_plan
from src.agent.dispatch import run_named_tool
from src.agent.intent import extract_intent


class BrainControlTests(unittest.TestCase):
    def test_tool_name_strips_wrapper_suffix(self):
        self.assertEqual(canonical_tool_name("web_search_tool"), "web_search")
        self.assertEqual(canonical_tool_name("set_trip_context"), "set_trip_context")

    def test_set_trip_context_is_model_interpretation(self):
        _, patch = run_named_tool(
            "set_trip_context",
            {
                "destination": "Belagavi",
                "also_known_as": "Belgaum, Belgaon",
                "state": "Karnataka",
                "climate_zone": "peninsular",
                "start_date": "2026-02-20",
                "end_date": "2026-02-23",
                "needs_trip_plan": True,
                "needs_alternatives": True,
            },
            {"user_request": "trip to Belgaon", "destination": "Belgaon"},
        )
        self.assertEqual(patch["destination"], "Belagavi")
        self.assertIn("Belgaon", patch["place_names"])
        self.assertEqual(patch["dest_state"], "Karnataka")
        self.assertEqual(patch["climate_zone"], "peninsular")
        self.assertTrue(patch["intent"]["needs_trip_plan"])

    def test_party_type_stored_from_brain(self):
        _, patch = run_named_tool(
            "set_trip_context",
            {"destination": "Hampi", "party_type": "family"},
            {},
        )
        self.assertEqual(patch["party_type"], "family")

    def test_nearby_from_events_not_only_annual_high(self):
        from src.agent.research import nearby_from_research

        quiet = nearby_from_research(
            {"crowd_levels": {"X": {"level": "LOW"}}, "period_pressure": {}, "web_context": {}}
        )
        self.assertFalse(quiet["suggest_nearby"])
        busy = nearby_from_research(
            {
                "crowd_levels": {"X": {"level": "LOW"}},
                "period_pressure": {"footfall_direction": "similar", "season": "dry_cool_season"},
                "web_context": {"events": [{"title": "local festival"}]},
            }
        )
        self.assertTrue(busy["suggest_nearby"])
        self.assertTrue(any("events" in r for r in busy["reasons"]))

    def test_offline_answer_falls_back_to_template(self):
        from unittest.mock import patch

        from src.agent.compose import compose_answer

        with patch.dict("os.environ", {"OPENAI_API_KEY": "", "TOURISM_LLM_BRAIN": "0"}, clear=False):
            text = compose_answer(
                {
                    "destination": "Taj Mahal",
                    "forecasts": {
                        "Taj Mahal": {
                            "found": True,
                            "destination": "Taj Mahal",
                            "predicted_visitors": 6909849,
                            "forecast_period": "2025-26",
                            "forecast_method": "persistence",
                        }
                    },
                    "crowd_levels": {},
                    "ranked_alternatives": [],
                    "tool_trace": [],
                }
            )
        self.assertIn("CROWD FORECAST", text)
        self.assertIn("6,909,849", text)

    def test_forecast_refuses_unknown_name(self):
        payload, patch = run_named_tool(
            "forecast_crowd",
            {"destination": "Belgaon"},
            {"forecasts": {}, "decisions": []},
        )
        self.assertFalse(payload.get("found"))
        self.assertIn("Belgaon", patch["forecasts"])

    def test_offline_taj_plan_still_forecasts(self):
        parsed = extract_intent("How crowded is Taj Mahal expected to be?")
        tools = fallback_tool_plan({**parsed, "tool_trace": []})
        self.assertIn("forecast_crowd", tools)

    def test_coerce_user_request_from_messages(self):
        from unittest.mock import patch

        from langchain_core.messages import HumanMessage

        from src.agent.graph import coerce_user_request, parse_intent

        req = coerce_user_request({"messages": [HumanMessage(content="Going to Hampi")]})
        self.assertIn("Hampi", req)
        with patch("src.agent.graph._llm_enabled", return_value=True):
            out = parse_intent({"messages": [HumanMessage(content="Belgaon with family")]})
        self.assertIn("Belgaon", out.get("user_request") or "")

    def test_live_parse_does_not_lock_place_before_openai(self):
        from unittest.mock import patch

        from src.agent.graph import parse_intent

        with patch("src.agent.graph._llm_enabled", return_value=True):
            out = parse_intent({"user_request": "Belgaon trip 20 to 23 Feb 2026 crowd and nearby"})
        self.assertIsNone(out.get("destination"))
        self.assertEqual(out.get("intent"), {})
