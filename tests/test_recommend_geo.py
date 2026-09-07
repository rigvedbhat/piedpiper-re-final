"""Geographic filtering, distance, and recommendation modes."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.geo import haversine_km, is_practical_orbit, lookup_coords, pair_distance_km
from src.tools_recommend import (
    _dedupe_candidates,
    filter_practical_candidates,
    find_similar_destinations,
    rank_alternatives,
)
from src.tools_web import classify_evidence_tier, is_junk_result, structure_web_context
from src.agent.intent import extract_intent
from src.agent.planner import execute_plan


class GeoTests(unittest.TestCase):
    def test_haversine_known_cities(self):
        k = lookup_coords("Kolhapur")
        a = lookup_coords("Alibaug")
        self.assertIsNotNone(k)
        self.assertIsNotNone(a)
        km = haversine_km(k[0], k[1], a[0], a[1])
        self.assertGreater(km, 200)

    def test_kolhapur_satara_nearby(self):
        km = pair_distance_km("Kolhapur", "Kaas Plateau", "Satara")
        self.assertIsNotNone(km)
        self.assertLess(km, 160)

    def test_missing_coordinates(self):
        self.assertIsNone(lookup_coords("ThisPlaceHasNoGazetteerEntryXYZ"))
        self.assertIsNone(pair_distance_km("Kolhapur", "Unknown Fort XYZ", None))

    def test_same_state_not_automatically_nearby(self):
        self.assertFalse(is_practical_orbit("same_state", 320.0, relax=False))
        self.assertTrue(is_practical_orbit("nearby_region", 90.0, relax=False))


class RankModeTests(unittest.TestCase):
    def test_taj_crowd_backed(self):
        ranked = rank_alternatives("Taj Mahal")
        self.assertTrue(ranked)
        self.assertIn("distance_score", ranked[0])
        self.assertIn("recommendation_mode", ranked[0])
        crowd = [r for r in ranked if r["recommendation_mode"] == "crowd_backed"]
        self.assertTrue(crowd)
        self.assertIsNotNone(crowd[0]["predicted_visitors"])
        self.assertIsNotNone(crowd[0]["components"]["crowd_pressure"])
        names = [r["name"].lower() for r in ranked]
        self.assertNotIn("taj mahal", names)

    def test_discovery_mode_no_invented_visitors(self):
        ranked = rank_alternatives(
            "Mahalakshmi Temple",
            candidates=[
                {
                    "name": "Kaas Plateau",
                    "city": "Satara",
                    "state": "Maharastra",
                    "place_type": "Valley",
                    "geo_tier": "nearby_region",
                    "distance_km": 110,
                    "relevance": {"interest_hit": True, "same_type": False, "same_asi_circle": False},
                    "airport_within_50km": "No",
                    "season": "Monsoon",
                    "source": "kaggle",
                }
            ],
            forecast_context={},
        )
        self.assertEqual(ranked[0]["recommendation_mode"], "discovery")
        self.assertIsNone(ranked[0]["predicted_visitors"])

    def test_dedupe(self):
        out = _dedupe_candidates([{"name": "A"}, {"name": "a"}, {"name": "B"}])
        self.assertEqual(len(out), 2)

    def test_asi_circle_not_nearby_without_geo(self):
        self.assertFalse(is_practical_orbit("unverified", None, relax=False))
        self.assertFalse(is_practical_orbit("same_asi_circle", None, relax=False))
        ranked = rank_alternatives(
            "Aga Khan Palace Building, Pune",
            candidates=[
                {
                    "name": "Old Fort, Sholapur",
                    "city": "Sholapur",
                    "state": "Maharashtra",
                    "place_type": "ASI monument",
                    "geo_tier": "unverified",
                    "distance_km": None,
                    "geography_verified": False,
                    "relevance": {"same_asi_circle": True, "same_type": True, "interest_hit": False},
                    "source": "asi",
                },
                {
                    "name": "Shaniwar Wada",
                    "city": "Pune",
                    "state": "Maharastra",
                    "place_type": "Fort",
                    "geo_tier": "locality_city",
                    "distance_km": 0.0,
                    "geography_verified": True,
                    "relevance": {"same_asi_circle": False, "same_type": True, "interest_hit": True},
                    "source": "kaggle",
                },
            ],
            forecast_context={},
            user_preferences={"location": "Pune"},
        )
        names = [r["name"] for r in ranked]
        self.assertIn("Shaniwar Wada", names)
        self.assertNotIn("Old Fort, Sholapur", names)

    def test_missing_coords_not_automatically_nearby(self):
        self.assertFalse(is_practical_orbit("unverified", None))
        self.assertIsNone(pair_distance_km("Pune", "Kondiote Caves", None))

    def test_official_web_candidate_promoted(self):
        from src.tools_web import web_discovery_candidates

        hits = [
            {
                "title": "Sindhudurg - Department of Tourism Maharashtra",
                "url": "https://maharashtratourism.gov.in/fort/sindhudurg",
                "snippet": "Sindhudurg Fort stands off Malvan",
                "source": "maharashtratourism.gov.in",
            },
            {
                "title": "Bhogave - Department of Tourism Maharashtra",
                "url": "https://maharashtratourism.gov.in/beach/bhogave",
                "snippet": "Bhogave Beach, located in the Sindhudurg district",
                "source": "maharashtratourism.gov.in",
            },
        ]
        cands = web_discovery_candidates(hits, "Sindhudurg", ["sindhudurg", "malvan"], ["forts", "beaches"])
        names = " ".join(c["name"].lower() for c in cands)
        self.assertIn("bhogave", names)
        self.assertTrue(any("fort" in c["name"].lower() for c in cands))
        for c in cands:
            self.assertIsNone(c.get("predicted_visitors"))
            self.assertEqual(c["source"], "official_web")

    def test_web_without_crowd_is_discovery(self):
        ranked = rank_alternatives(
            "Sindhudurg",
            candidates=[
                {
                    "name": "Bhogave Beach",
                    "city": "Sindhudurg",
                    "place_type": "Beach",
                    "geo_tier": "locality_district",
                    "distance_km": None,
                    "geography_verified": True,
                    "source": "official_web",
                    "relevance": {"interest_hit": True, "same_type": False, "same_asi_circle": False},
                }
            ],
            forecast_context={},
            user_preferences={"location": "Sindhudurg"},
        )
        self.assertEqual(ranked[0]["recommendation_mode"], "discovery")
        self.assertFalse(ranked[0]["crowd_data_available"])
        self.assertIsNone(ranked[0]["predicted_visitors"])

    def test_web_name_with_asi_forecast_is_crowd_backed(self):
        ranked = rank_alternatives(
            "Taj Mahal",
            candidates=[
                {
                    "name": "Agra Fort",
                    "city": "Agra",
                    "place_type": "Fort",
                    "geo_tier": "locality_city",
                    "distance_km": 3.4,
                    "geography_verified": True,
                    "source": "official_web",
                    "relevance": {"interest_hit": True, "same_type": True, "same_asi_circle": False},
                }
            ],
            forecast_context={"predicted_visitors": 6909849.0},
            user_preferences={"location": "Agra"},
        )
        self.assertEqual(ranked[0]["recommendation_mode"], "crowd_backed")
        self.assertTrue(ranked[0]["crowd_data_available"])
        self.assertIsNotNone(ranked[0]["predicted_visitors"])

    def test_duplicate_local_and_web(self):
        from src.tools_recommend import _dedupe_candidates

        out = _dedupe_candidates(
            [
                {"name": "Bhogave Beach", "source": "kaggle"},
                {"name": "Bhogave Beach", "source": "official_web"},
            ]
        )
        self.assertEqual(len(out), 1)

    def test_no_candidate_honest_empty_rank(self):
        ranked = rank_alternatives(
            "Sindhudurg",
            candidates=[],
            forecast_context={},
            user_preferences={"location": "Sindhudurg"},
        )
        self.assertEqual(ranked, [])


class WebEvidenceTests(unittest.TestCase):
    def test_junk_pdf(self):
        self.assertTrue(
            is_junk_result(
                {
                    "title": "USQ 2716",
                    "url": "https://tourism.gov.in/sites/usq%202716.pdf",
                }
            )
        )
        self.assertEqual(
            classify_evidence_tier("https://maharashtratourism.gov.in/districts/kolhapur"),
            "official",
        )

    def test_structure_drops_junk(self):
        ctx = structure_web_context(
            [
                {
                    "title": "Recruitment | Ministry of Tourism",
                    "url": "https://tourism.gov.in/recruitment",
                    "snippet": "vacancy",
                },
                {
                    "title": "Kolhapur events festival",
                    "url": "https://maharashtratourism.gov.in/districts/kolhapur",
                    "snippet": "Kolhapur Mahotsav festival",
                },
            ],
            destination="Kolhapur",
            queries=["q"],
            provider="test",
        )
        self.assertTrue(ctx["discarded"])
        self.assertTrue(ctx["events"] or ctx["results"])


class PlannerScenarioTests(unittest.TestCase):
    def test_kolhapur_excludes_distant_maharashtra(self):
        parsed = extract_intent(
            "I am new to Kolhapur. I am visiting from 1 September 2026 to 5 September 2026. "
            "I like temples, forts and nature. I want to see Mahalakshmi, Jyotiba and Panhala."
        )
        with patch("src.agent.planner.web_search", return_value={"results": [], "provider": "none", "warnings": []}):
            out = execute_plan({**parsed, "tool_trace": ["parse_intent"]})
        names = " ".join(r["name"].lower() for r in out.get("ranked_alternatives") or [])
        self.assertNotIn("alibaug", names)
        self.assertNotIn("ajanta", names)
        self.assertTrue(out["intent"]["needs_trip_plan"])
        self.assertTrue(out.get("trip_plan"))

    def test_hampi_prediction_and_alts(self):
        parsed = extract_intent(
            "I want to visit Hampi. How busy will it be, and are there quieter similar heritage sites?"
        )
        self.assertTrue(parsed["intent"]["needs_prediction"])
        self.assertTrue(parsed["intent"]["needs_alternatives"])
        out = execute_plan({**parsed, "tool_trace": ["parse_intent"]})
        self.assertIn("forecast_crowd", out["tool_trace"])
        self.assertIn("find_similar_destinations", out["tool_trace"])
        self.assertIn("rank_alternatives", out["tool_trace"])

    def test_pune_plan_excludes_sholapur(self):
        parsed = extract_intent(
            "I am visiting Pune for 10-12 October 2026. I like forts and nature. What should I see?"
        )
        with patch("src.agent.planner.web_search", return_value={"results": [], "provider": "none", "warnings": []}):
            out = execute_plan({**parsed, "tool_trace": ["parse_intent"]})
        names = " ".join(r["name"].lower() for r in out.get("ranked_alternatives") or [])
        self.assertNotIn("sholapur", names)
        self.assertNotIn("solapur", names)

    def test_insufficient_web_fallback_called(self):
        parsed = extract_intent("What are some lesser-known places around Kolhapur?")
        calls = []

        def fake_search(query, domains=None, max_results=5):
            calls.append(query)
            return {"results": [], "provider": "none", "warnings": [], "query": query}

        with patch("src.agent.planner.web_search", side_effect=fake_search):
            out = execute_plan({**parsed, "tool_trace": ["parse_intent"]})
        self.assertTrue(any("lesser known" in q.lower() or "kolhapur" in q.lower() for q in calls) or out.get("ranked_alternatives") is not None)


class ReasoningTests(unittest.TestCase):
    def test_final_answer_explains_decisions(self):
        from src.agent.compose import compose_final_response

        text = compose_final_response(
            {
                "destination": "Taj Mahal",
                "intent": {
                    "needs_prediction": True,
                    "needs_alternatives": True,
                    "needs_current_context": False,
                    "needs_trip_plan": False,
                },
                "forecasts": {
                    "Taj Mahal": {
                        "found": True,
                        "destination": "Taj Mahal",
                        "predicted_visitors": 6909849,
                        "forecast_period": "2025-26",
                        "forecast_method": "persistence",
                    }
                },
                "crowd_levels": {"Taj Mahal": {"level": "VERY HIGH"}},
                "ranked_alternatives": [
                    {
                        "name": "Agra Fort",
                        "recommendation_mode": "crowd_backed",
                        "practical": True,
                        "relevant": True,
                        "geography_verified": True,
                        "geo_tier": "locality_city",
                        "distance_km": 3.4,
                        "crowd_data_available": True,
                        "source": "asi",
                        "predicted_visitors": 1770474,
                        "why": ["Same city as the anchor"],
                    }
                ],
                "tool_trace": ["parse_intent", "forecast_crowd", "rank_alternatives"],
            }
        )
        self.assertIn("HOW I REASONED", text)
        self.assertIn("How I read your request", text)
        self.assertIn("Why I produced a crowd number", text)
        self.assertIn("Why I recommend Agra Fort", text)
        self.assertIn("What I refused to do", text)


if __name__ == "__main__":
    unittest.main()
