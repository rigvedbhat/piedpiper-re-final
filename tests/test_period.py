"""Travel-window pressure does not invent a new visitor count."""

from __future__ import annotations

import unittest

from src.agent.brain import fallback_tool_plan
from src.agent.intent import extract_intent
from src.tools_period import assess_period_pressure


class PeriodPressureTests(unittest.TestCase):
    def test_no_dates_unknown(self):
        out = assess_period_pressure("Taj Mahal", None, None, crowd_level="VERY HIGH")
        self.assertEqual(out["footfall_direction"], "unknown")
        self.assertTrue(out["does_not_change_annual_count"])

    def test_winter_weekend_higher(self):
        out = assess_period_pressure(
            "Taj Mahal",
            "2026-01-24",
            "2026-01-26",
            crowd_level="VERY HIGH",
        )
        self.assertTrue(out["found"])
        self.assertEqual(out["footfall_direction"], "likely_higher")
        self.assertIn("2026-01-26", out["holiday_dates"])

    def test_monsoon_week_not_a_new_count(self):
        out = assess_period_pressure(
            "Mumbai",
            "2022-07-04",
            "2022-07-05",
            crowd_level=None,
        )
        self.assertTrue(out["does_not_change_annual_count"])
        self.assertIsNone(out.get("predicted_visitors"))

    def test_taj_undated_fallback_tools_no_web(self):
        parsed = extract_intent("How crowded is Taj Mahal expected to be?")
        tools = fallback_tool_plan({**parsed, "tool_trace": []})
        self.assertIn("forecast_crowd", tools)
        self.assertNotIn("web_search", tools)

    def test_karnataka_february_is_not_north_india_peak(self):
        out = assess_period_pressure(
            "Belagavi",
            "2026-02-20",
            "2026-02-23",
            dest_state="Karnataka",
            climate_zone="peninsular",
        )
        self.assertEqual(out["season"], "dry_cool_season")
        self.assertNotIn("peak_winter_tourism", out["season"])
        self.assertTrue(all("north-India heritage" not in r for r in out["reasons"]))
