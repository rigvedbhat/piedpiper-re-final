"""Critic loop does not change forecast or ranking formulas."""

from __future__ import annotations

import unittest

from src.agent.critic import critic_has_gap, destination_looks_junk
from src.agent.intent import extract_intent, resolve_intent


class CriticTests(unittest.TestCase):
    def test_instead_phrase_is_junk(self):
        self.assertTrue(destination_looks_junk("Should I Visit Instead"))
        self.assertFalse(destination_looks_junk("Pratapgad"))

    def test_resolve_without_llm_keeps_keywords(self):
        parsed = resolve_intent(
            "Will Pratapgad fort be crowded, and what should I visit instead?",
            use_llm=False,
        )
        self.assertEqual(parsed["destination"], "Pratapgad")

    def test_gap_when_alts_missing_and_no_web(self):
        parsed = extract_intent("Will Pratapgad fort be crowded, and what should I visit instead?")
        state = {
            **parsed,
            "ranked_alternatives": [],
            "tool_trace": ["parse_intent", "forecast_crowd"],
        }
        self.assertTrue(critic_has_gap(state))
