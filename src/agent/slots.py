"""Slot extraction is now handled by the brain's set_trip_context tool.
This module is kept for backward compatibility but no longer makes a separate LLM call."""

from __future__ import annotations


def llm_extract_slots(text: str) -> dict | None:
    """No-op. The Gemini brain extracts slots via set_trip_context in its tool loop.
    Keeping this function so resolve_intent() doesn't break."""
    return None
