"""LangGraph Server / Agent Chat UI adapter.

Does not reimplement tourism logic. Every turn calls run_agent()
(the same graph as python demo.py).
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from src.agent.graph import run_agent

GRAPH_ID = "tourism"


def _message_text(message: BaseMessage | dict) -> str:
    if isinstance(message, dict):
        content = message.get("content") or ""
        if isinstance(content, str):
            return content
        return str(content)
    content = getattr(message, "content", "") or ""
    if not isinstance(content, str):
        content = str(content)
    return content


def _message_role(message: BaseMessage | dict) -> str:
    if isinstance(message, dict):
        role = (message.get("type") or message.get("role") or "").lower()
        if role in {"human", "user"}:
            return "traveler"
        if role in {"ai", "assistant", "ai_message"}:
            return "advisor"
        return ""
    if isinstance(message, HumanMessage):
        return "traveler"
    if isinstance(message, AIMessage):
        return "advisor"
    msg_type = (getattr(message, "type", "") or "").lower()
    if msg_type in {"human", "user"}:
        return "traveler"
    if msg_type in {"ai", "assistant"}:
        return "advisor"
    return ""


def conversation_transcript(messages: list) -> str:
    """Full back-and-forth for the brain (traveler + advisor)."""
    lines: list[str] = []
    for msg in (messages or [])[-16:]:
        text = _message_text(message=msg).strip()
        if not text:
            continue
        role = _message_role(message=msg)
        if role == "traveler":
            lines.append(f"Traveler: {text}")
        elif role == "advisor":
            lines.append(f"Advisor: {text}")
    return "\n".join(lines)


def conversation_to_user_request(messages: list) -> str:
    """Latest traveler line, with earlier turns as conversation context."""
    transcript = conversation_transcript(messages)
    humans: list[str] = []
    for msg in messages or []:
        if _message_role(message=msg) == "traveler":
            text = _message_text(message=msg).strip()
            if text:
                humans.append(text)
    latest = humans[-1] if humans else ""
    if not transcript:
        return latest
    return f"{transcript}\n\nLatest traveler message:\n{latest}"


def format_chat_answer(raw: str) -> str:
    """Chat bubble. Returns conversational response directly."""
    text = (raw or "").strip()
    return text if text else "(no response from tourism agent)"


def tool_event_messages(events: list[dict]) -> list[BaseMessage]:
    """Replay the inner brain's actual tool calls in Agent Chat.

    The adapter does not choose or run tools itself. It only makes the
    brain-selected calls and the facts returned by Python visible to the UI.
    """
    messages: list[BaseMessage] = []
    for index, event in enumerate(events or [], start=1):
        name = str(event.get("name") or "unknown_tool")
        call_id = f"tourism-tool-{index}"
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        observation = event.get("observation")
        messages.append(
            AIMessage(
                content="",
                tool_calls=[{"name": name, "args": args, "id": call_id}],
            )
        )
        messages.append(
            ToolMessage(
                name=name,
                tool_call_id=call_id,
                content=json.dumps(observation, default=str),
            )
        )
    return messages


def tourism_turn(state: MessagesState) -> dict:
    """One chat turn → tourism agent. Missing slots become a conversational ask."""
    messages = state.get("messages") or []
    transcript = conversation_transcript(messages)
    request = conversation_to_user_request(messages)
    if not request.strip():
        request = "(The traveler has not said anything yet.)"
        transcript = request
    try:
        result = run_agent(request, conversation=transcript)
        content = format_chat_answer(result.get("final_response") or "")
        events = tool_event_messages(result.get("tool_events") or [])
    except Exception as exc:
        content = (
            "I couldn't finish that just now. Nothing was invented as a crowd number.\n\n"
            f"{type(exc).__name__}: {exc}"
        )
        events = []
    return {"messages": [*events, AIMessage(content=content)]}


def build_chat_graph():
    g = StateGraph(MessagesState)
    g.add_node("tourism_agent", tourism_turn)
    g.add_edge(START, "tourism_agent")
    g.add_edge("tourism_agent", END)
    return g.compile()


graph = build_chat_graph()
