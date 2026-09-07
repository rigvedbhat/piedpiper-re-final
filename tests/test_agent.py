"""Agent graph tests that do not require OpenAI for most cases."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

from src.agent.graph import ingest_tool_results, route_after_agent
from src.agent.optimize import optimize_trip
from src.agent.state import TourismState
from src.tools_local import forecast_crowd, get_historical_footfall, search_destinations
from src.tools_recommend import rank_alternatives
from src.tools_web import web_search


class OptimizeTests(unittest.TestCase):
    def test_kolhapur_length(self):
        plan = optimize_trip(
            "2026-09-01",
            "2026-09-05",
            ["Mahalakshmi", "Jyotiba", "Panhala"],
            [{"name": "Ranked Orbit A"}, {"name": "Ranked Orbit B"}],
            {},
        )
        self.assertEqual(len(plan), 5)
        self.assertEqual(plan[0]["type"], "anchor")
        self.assertEqual(plan[3]["type"], "orbit")


class RoutingTests(unittest.TestCase):
    def test_tools_when_tool_calls(self):
        state: TourismState = {
            "messages": [AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "1"}])],
            "is_multi_day": False,
        }
        self.assertEqual(route_after_agent(state), "tools")

    def test_optimize_when_multi_day_and_done(self):
        state: TourismState = {
            "messages": [AIMessage(content="done")],
            "is_multi_day": True,
        }
        self.assertEqual(route_after_agent(state), "optimize_trip")

    def test_ingest_forecast(self):
        payload = forecast_crowd("Taj Mahal", "2025-26")
        state: TourismState = {
            "messages": [
                ToolMessage(
                    content=json.dumps(payload),
                    name="forecast_crowd_tool",
                    tool_call_id="1",
                )
            ],
            "tool_trace": [],
        }
        out = ingest_tool_results(state)
        self.assertEqual(out["forecasts"]["Taj Mahal"]["forecast_method"], "persistence")
        self.assertEqual(out["forecasts"]["Taj Mahal"]["predicted_visitors"], payload["predicted_visitors"])


class LocalPipelineCases(unittest.TestCase):
    def test_case1_taj_prediction(self):
        h = get_historical_footfall("Taj Mahal")
        fc = forecast_crowd("Taj Mahal", "2025-26")
        self.assertEqual(fc["predicted_visitors"], h["observations"][-1]["total_visitors"])

    def test_case2_taj_alternatives_not_hardcoded(self):
        ranked = rank_alternatives("Taj Mahal")
        names = [r["name"] for r in ranked[:3]]
        self.assertTrue(ranked)
        self.assertNotIn("Taj Mahal", names)

    def test_case4_unknown(self):
        h = get_historical_footfall("ThisIsNotARealMonumentXYZ")
        self.assertFalse(h["found"])

    def test_case5_search_still_annual(self):
        fc = forecast_crowd("Taj Mahal")
        self.assertEqual(fc["data_granularity"], "annual")

    def test_case6_web_search_failure_shape(self):
        with patch.dict("os.environ", {"TAVILY_API_KEY": ""}, clear=False):
            with patch("src.tools_web._wikipedia_search", side_effect=OSError("offline")):
                with self.assertRaises(OSError):
                    web_search("Taj Mahal")

    def test_case7_unknown_has_no_kaggle_required(self):
        hits = search_destinations("ThisIsNotARealMonumentXYZ")
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
