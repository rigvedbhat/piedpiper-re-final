"""Web discovery should not treat other-state pages as nearby."""

from __future__ import annotations

import unittest

from src.tools_web import (
    _normalize_attraction_name,
    structure_web_context,
    web_discovery_candidates,
)


class WebDiscoveryCleanupTests(unittest.TestCase):
    def test_drops_unrelated_state_page(self):
        ctx = structure_web_context(
            [
                {
                    "title": "Hassan - Best Places To Visit In Hassan District | Karnataka Tourism",
                    "url": "https://karnatakatourism.org/destinations/hassan",
                    "snippet": "Hoysala Empire Belur Halebeedu",
                    "source": "karnatakatourism.org",
                },
                {
                    "title": "Sangli - Department of Tourism Maharashtra",
                    "url": "https://maharashtratourism.gov.in/districts/sangli",
                    "snippet": "Kopeshwar Temple and Sagareshwar Sanctuary near Sangli",
                    "source": "maharashtratourism.gov.in",
                },
            ],
            destination="Sangli",
            queries=["Sangli tourism"],
            provider="tavily",
            nearby_tokens=["sangli", "miraj"],
        )
        urls = " ".join(h.get("url") or "" for h in ctx.get("results") or [])
        self.assertNotIn("hassan", urls.lower())
        self.assertIn("sangli", urls.lower())

    def test_cleans_duplicate_and_facebook_names(self):
        self.assertEqual(
            _normalize_attraction_name("India Kopeshwar Temple Kopeshwar Temple"),
            "Kopeshwar Temple",
        )
        self.assertNotIn(
            "facebook",
            _normalize_attraction_name("Facebook Zoom Sagareshwar Sanctuary").lower(),
        )
        hits = [
            {
                "title": "Sagareshwar Sanctuary - Sangli",
                "url": "https://maharashtratourism.gov.in/sanctuary/sagareshwar",
                "snippet": "Sagareshwar Sanctuary in Sangli district",
                "source": "maharashtratourism.gov.in",
            }
        ]
        cands = web_discovery_candidates(hits, "Sangli", ["sangli"], None)
        names = [c["name"].lower() for c in cands]
        self.assertTrue(all("facebook" not in n for n in names))
        self.assertTrue(any("sagareshwar" in n for n in names))
        for c in cands:
            self.assertNotEqual(c.get("distance_km"), 0.0)

    def test_wrong_state_portal_dropped_for_any_origin(self):
        ctx = structure_web_context(
            [
                {
                    "title": "Hassan district",
                    "url": "https://karnatakatourism.org/destinations/hassan",
                    "snippet": "Hassan Karnataka",
                    "source": "karnatakatourism.org",
                }
            ],
            destination="Pune",
            queries=["Pune tourism"],
            provider="tavily",
            nearby_tokens=["pune"],
            dest_state="Maharashtra",
        )
        self.assertEqual(ctx.get("results") or [], [])

    def test_foreign_gazetteer_city_not_a_nearby_candidate(self):
        hits = [
            {
                "title": "Hassan - Department of Tourism",
                "url": "https://maharashtratourism.gov.in/districts/sangli",
                "snippet": "Hassan is listed among destinations while talking about Sangli",
                "source": "maharashtratourism.gov.in",
            }
        ]
        cands = web_discovery_candidates(
            hits, "Sangli", ["sangli"], None, dest_state="Maharashtra"
        )
        names = [c["name"].lower() for c in cands]
        self.assertTrue(all("hassan" != n for n in names))
        self.assertTrue(all(not n.startswith("hassan ") for n in names))
