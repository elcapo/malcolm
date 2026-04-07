"""Canonical models for normalized LLM request/response representation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """A normalized tool/function call."""

    name: str
    arguments: str = ""
    id: str = ""


@dataclass
class Message:
    """A single message in a conversation, format-agnostic."""

    role: str
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_result: str | None = None
    raw: dict = field(default_factory=dict)


@dataclass
class SessionGroup:
    """A group of related requests (a 'session')."""

    session_id: str
    record_ids: list[str] = field(default_factory=list)
    model: str = ""
    earliest_timestamp: str = ""
    latest_timestamp: str = ""
    request_count: int = 0


@dataclass
class Conversation:
    """A normalized request+response pair."""

    messages: list[Message] = field(default_factory=list)
    model: str = ""
    timestamp: str = ""
    stream: bool = False
    status_code: int | None = None
    duration_ms: float | None = None
    error: str | None = None
