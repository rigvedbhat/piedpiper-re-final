"""Intent flags and planner routing without OpenAI."""

from __future__ import annotations

import unittest

from src.agent.intent import extract_intent
from src.agent.planner import execute_plan


class IntentTests(unittest.TestCase):
    def test_taj_prediction_only(self):
        i = extract_intent("How crowded is Taj Mahal expected to be?")
        self.assertTrue(i["intent"]["needs_prediction"])
        self.assertFalse(i["intent"]["needs_alternatives"])
        self.assertFalse(i["intent"]["needs_trip_plan"])
        self.assertEqual(i["destination"], "Taj Mahal")

    def test_taj_instead(self):
        i = extract_intent("Will Taj Mahal be crowded, and what should I visit instead?")
        self.assertTrue(i["intent"]["needs_prediction"])
        self.assertTrue(i["intent"]["needs_alternatives"])

    def test_kolhapur_trip(self):
        i = extract_intent(
            "I am new to Kolhapur. I am visiting from 1 September 2026 to 5 September 2026. "
            "I like temples, forts and nature. I want to see Mahalakshmi, Jyotiba and Panhala."
        )
        self.assertEqual(i["destination"], "Kolhapur")
        self.assertEqual(i["start_date"], "2026-09-01")
        self.assertEqual(i["end_date"], "2026-09-05")
        self.assertTrue(i["intent"]["needs_trip_plan"])
        self.assertTrue(i["intent"]["needs_alternatives"])
        self.assertTrue(i["intent"]["needs_current_context"])
        self.assertEqual(i["must_visit"], ["Mahalakshmi", "Jyotiba", "Panhala"])

    def test_happening(self):
        i = extract_intent("What's happening in Kolhapur during Sep 1-5?")
        self.assertTrue(i["intent"]["needs_current_context"])
        self.assertFalse(i["intent"]["needs_trip_plan"])
        self.assertFalse(i["intent"]["needs_alternatives"])

    def test_who(self):
        i = extract_intent("Who is Zorblax?")
        self.assertFalse(i["intent"]["needs_prediction"])
        self.assertEqual(i["destination"], "Zorblax")

    def test_pratapgad_not_instead_phrase(self):
        i = extract_intent("Will Pratapgad fort be crowded, and what should I visit instead?")
        self.assertEqual(i["destination"], "Pratapgad")
        self.assertTrue(i["intent"]["needs_prediction"])
        self.assertTrue(i["intent"]["needs_alternatives"])
        self.assertNotIn("Instead", i["destination"] or "")

    def test_mumbai_trip(self):
        i = extract_intent("I am new to Mumbai. I am visiting from July 1 to July 5, 2022..")
        self.assertEqual(i["destination"], "Mumbai")
        self.assertEqual(i["start_date"], "2022-07-01")
        self.assertTrue(i["intent"]["needs_trip_plan"])

    def test_kutub_alias(self):
        i = extract_intent("How crowded will Kutub Minar be?")
        self.assertEqual(i["destination"], "Qutub Minar")

    def test_generic_city_trip_with_ordinal_dates(self):
        i = extract_intent(
            "I am visiting Udupi from 5th September 2026 to 7th September 2026. "
            "Can you tell me the best spots to visit?"
        )
        self.assertEqual(i["destination"], "Udupi")
        self.assertEqual(i["start_date"], "2026-09-05")
        self.assertEqual(i["end_date"], "2026-09-07")
        self.assertTrue(i["intent"]["needs_trip_plan"])


class PlannerRouteTests(unittest.TestCase):
    def test_taj_prediction_tools(self):
        parsed = extract_intent("How crowded is Taj Mahal expected to be?")
        out = execute_plan({**parsed, "tool_trace": ["parse_intent"]})
        self.assertIn("forecast_crowd", out["tool_trace"])
        self.assertIn("classify_crowd", out["tool_trace"])
        self.assertNotIn("find_similar_destinations", out["tool_trace"])
        self.assertNotIn("rank_alternatives", out["tool_trace"])
        self.assertNotIn("web_search", out["tool_trace"])

    def test_unknown_town_searches_web_for_famous_places(self):
        from unittest.mock import patch

        from src.agent.planner import execute_plan

        parsed = extract_intent("I want to visit Sangli.")
        self.assertEqual(parsed["destination"], "Sangli")
        with patch(
            "src.agent.planner.web_search",
            return_value={"provider": "mock", "results": []},
        ) as mocked:
            out = execute_plan({**parsed, "tool_trace": ["parse_intent"]})
        queries = " ".join(str(c.args[0]) for c in mocked.call_args_list).lower()
        self.assertIn("sangli", queries)
        self.assertTrue("famous" in queries or "tourist" in queries)
        self.assertIn("web_search", out["tool_trace"])
        self.assertTrue(out["intent"].get("needs_alternatives"))

    def test_taj_instead_tools(self):
        parsed = extract_intent("Will Taj Mahal be crowded, and what should I visit instead?")
        out = execute_plan({**parsed, "tool_trace": ["parse_intent"]})
        self.assertIn("forecast_crowd", out["tool_trace"])
        self.assertIn("find_similar_destinations", out["tool_trace"])
        self.assertIn("rank_alternatives", out["tool_trace"])


if __name__ == "__main__":
    unittest.main()
