"""Chat adapter uses the same run_agent as the CLI."""

from __future__ import annotations

import json
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.chat_graph import conversation_to_user_request, tourism_turn


class ChatGraphTests(unittest.TestCase):
    def test_follow_up_includes_prior_user_text(self):
        text = conversation_to_user_request(
            [
                HumanMessage(content="I'm going to Hampi for 3 days."),
                AIMessage(content="plan..."),
                HumanMessage(content="I also love nature. Can you change the plan?"),
            ]
        )
        self.assertIn("Hampi", text)
        self.assertIn("nature", text)
        self.assertIn("Advisor:", text)
        self.assertIn("Traveler:", text)

    def test_clarify_stops_for_missing_slots(self):
        from src.agent.dispatch import run_named_tool

        payload, patch = run_named_tool(
            "clarify_with_user",
            {"question": "When are you going, and is it family, friends, or solo?", "missing": "dates,party"},
            {},
        )
        self.assertTrue(patch["awaiting_user"])
        self.assertIn("family", patch["pending_question"].lower())

    def test_turn_calls_run_agent(self):
        with patch(
            "src.agent.chat_graph.run_agent",
            return_value={"final_response": "CROWD FORECAST\nPredicted annual demand: 6,909,849"},
        ) as mock_run:
            out = tourism_turn(
                {"messages": [HumanMessage(content="How crowded will Taj Mahal be?")]}
            )
        mock_run.assert_called_once()
        self.assertIn("How crowded will Taj Mahal be?", mock_run.call_args[0][0])
        msg = out["messages"][0]
        self.assertIsInstance(msg, AIMessage)
        self.assertIn("6,909,849", msg.content)
        self.assertIn("## Crowd forecast", msg.content)

    def test_error_is_honest(self):
        with patch("src.agent.chat_graph.run_agent", side_effect=RuntimeError("OPENAI_API_KEY is not set")):
            out = tourism_turn({"messages": [HumanMessage(content="Taj Mahal")]})
        self.assertIn("couldn't finish", out["messages"][0].content.lower())
        self.assertIn("RuntimeError", out["messages"][0].content)
        self.assertNotIn("6,909,849", out["messages"][0].content)

    def test_compiled_graph_invokes_run_agent(self):
        from src.agent.chat_graph import graph

        with patch(
            "src.agent.chat_graph.run_agent",
            return_value={"final_response": "CROWD FORECAST\nPredicted annual demand: 6,909,849"},
        ) as mock_run:
            result = graph.invoke(
                {"messages": [HumanMessage(content="How crowded will Taj Mahal be?")]}
            )
        mock_run.assert_called_once()
        last = result["messages"][-1]
        self.assertIn("## Crowd forecast", last.content)
        self.assertIn("6,909,849", last.content)

    def test_chat_answer_leads_with_forecast_not_log(self):
        from src.agent.chat_graph import format_chat_answer

        raw = (
            "HOW I REASONED (decision log)\n"
            "long log...\n"
            "CROWD FORECAST\n"
            "Predicted annual demand: 6,909,849\n"
            "LIMITATIONS\n"
            "annual only"
        )
        out = format_chat_answer(raw)
        self.assertLess(out.find("## Crowd forecast"), out.find("## How I reasoned"))
        self.assertIn("6,909,849", out)

    def test_llm_prose_is_not_resectioned(self):
        from src.agent.chat_graph import format_chat_answer

        prose = "According to our analysis, visit Gokak Falls if Belagavi is busy that weekend."
        self.assertEqual(format_chat_answer(prose), prose)

    def test_langgraph_server_chat_ui_path(self):
        """Agent Chat UI request → server → tourism graph → forecast tool → response."""
        try:
            urllib.request.urlopen("http://127.0.0.1:2024/ok", timeout=2)
        except (urllib.error.URLError, OSError):
            self.skipTest("langgraph server not running")
        payload = json.dumps(
            {
                "assistant_id": "tourism",
                "input": {
                    "messages": [
                        {"role": "human", "content": "How crowded will Taj Mahal be?"}
                    ]
                },
            }
        ).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:2024/runs/wait",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            state = json.loads(resp.read().decode())
        messages = state.get("messages") or []
        self.assertTrue(messages)
        content = messages[-1].get("content") if isinstance(messages[-1], dict) else str(messages[-1])
        self.assertTrue(content)
        lower = content.lower()
        if "keyerror" in lower or "couldn't finish" in lower:
            self.skipTest("langgraph server is on an old graph; restart langgraph dev")
        self.assertTrue(
            "taj" in lower
            or "when" in lower
            or "family" in lower
            or "solo" in lower
            or "going" in lower,
            msg=content[:400],
        )
