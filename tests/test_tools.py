"""Unit tests for local tools, ranking, and forecast wrappers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.tools_local import (
    classify_crowd,
    forecast_crowd,
    get_destination_profile,
    get_historical_footfall,
    search_destinations,
)
from src.tools_recommend import find_similar_destinations, rank_alternatives
from src.tools_web import web_search


class ToolTests(unittest.TestCase):
    def test_search_taj(self):
        hits = search_destinations("Taj Mahal")
        names = [h["name"] for h in hits]
        self.assertTrue(any("Taj Mahal" in n for n in names))

    def test_profile_taj(self):
        p = get_destination_profile("Taj Mahal")
        self.assertEqual(p["asi_name"], "Taj Mahal")
        self.assertTrue(p["has_asi_history"])

    def test_historical_taj(self):
        h = get_historical_footfall("Taj Mahal")
        self.assertTrue(h["found"])
        self.assertEqual(h["granularity"], "annual")
        self.assertGreaterEqual(len(h["observations"]), 4)
        self.assertIn(h["annual_direction"], {"increasing", "decreasing", "similar"})

    def test_forecast_persistence(self):
        h = get_historical_footfall("Taj Mahal")
        last = h["observations"][-1]["total_visitors"]
        fc = forecast_crowd("Taj Mahal", "2025-26")
        self.assertEqual(fc["forecast_method"], "persistence")
        self.assertEqual(fc["predicted_visitors"], last)

    def test_classify_not_occupancy(self):
        c = classify_crowd("Taj Mahal")
        self.assertIn(c["level"], {"LOW", "MODERATE", "HIGH", "VERY HIGH"})
        self.assertIn("not physical occupancy", c["disclaimer"].lower())

    def test_unknown_destination_history(self):
        h = get_historical_footfall("ThisIsNotARealMonumentXYZ")
        self.assertFalse(h["found"])

    def test_similar_excludes_self(self):
        alts = find_similar_destinations("Taj Mahal", interests=["heritage"])
        self.assertTrue(alts)
        self.assertFalse(any(a["name"].lower() == "taj mahal" for a in alts))

    def test_rank_has_components(self):
        ranked = rank_alternatives("Taj Mahal")
        self.assertTrue(ranked)
        top = ranked[0]
        self.assertIn("final_score", top)
        self.assertIn("similarity", top["components"])
        self.assertIn("crowd_pressure", top["components"])
        self.assertNotEqual(top["name"].lower(), "taj mahal")

    def test_web_search_structure_offline(self):
        fake = {
            "results": [
                {
                    "title": "Taj Mahal",
                    "url": "https://en.wikipedia.org/wiki/Taj_Mahal",
                    "source": "en.wikipedia.org",
                    "published": None,
                    "snippet": "mausoleum",
                }
            ]
        }
        with patch.dict("os.environ", {"TAVILY_API_KEY": ""}, clear=False):
            with patch("src.tools_web._wikipedia_search", return_value=fake):
                out = web_search("Taj Mahal")
        self.assertEqual(out["layer"], "FACTS")
        self.assertEqual(out["results"][0]["url"], "https://en.wikipedia.org/wiki/Taj_Mahal")


if __name__ == "__main__":
    unittest.main()
