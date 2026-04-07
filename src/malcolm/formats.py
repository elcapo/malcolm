"""Format detection, parsing, and normalization for LLM API request/response data.

Provides a dispatch mechanism that iterates over format-specific parsers
to convert raw stored data into canonical Message/Conversation objects.
"""

from __future__ import annotations

import json
from typing import Protocol

from malcolm.models import Conversation, Message, SessionGroup, ToolCall


# ---------------------------------------------------------------------------
# Parser protocol
# ---------------------------------------------------------------------------


class FormatParser(Protocol):
    """Interface that each format-specific parser must implement."""

    def can_parse_request(self, body: dict) -> bool: ...
    def parse_request_messages(self, body: dict) -> list[Message]: ...
    def can_parse_response(self, body: dict) -> bool: ...
    def parse_response(self, body: dict) -> Message | None: ...
    def assemble_chunks(self, chunks: list[dict]) -> Message | None: ...
    def extract_session_hint(self, body: dict, headers: dict | None = None) -> str | None: ...


# ---------------------------------------------------------------------------
# OpenAI parser
# ---------------------------------------------------------------------------


class OpenAIParser:
    """Parses OpenAI Chat Completions format."""

    def can_parse_request(self, body: dict) -> bool:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return False
        # OpenAI requests have messages with string content or list of
        # {"type": "text"/"image_url", ...} parts — no Anthropic-style
        # content blocks with "type": "tool_use"/"tool_result".
        first_content = messages[0].get("content")
        if isinstance(first_content, list):
            types = {b.get("type") for b in first_content if isinstance(b, dict)}
            if types & {"tool_use", "tool_result"}:
                return False
        return True

    def parse_request_messages(self, body: dict) -> list[Message]:
        result: list[Message] = []
        for msg in body.get("messages", []):
            result.append(self._parse_message(msg))
        return result

    def can_parse_response(self, body: dict) -> bool:
        return "choices" in body

    def parse_response(self, body: dict) -> Message | None:
        for choice in body.get("choices", []):
            msg = choice.get("message") or choice.get("delta")
            if not msg:
                continue
            return self._parse_message(msg)
        return None

    def assemble_chunks(self, chunks: list[dict]) -> Message | None:
        text_parts: list[str] = []
        tool_calls_by_index: dict[int, ToolCall] = {}

        for chunk in chunks:
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content")
                if content:
                    text_parts.append(content)

                for tc in delta.get("tool_calls", []):
                    idx = tc.get("index", 0)
                    if idx not in tool_calls_by_index:
                        tool_calls_by_index[idx] = ToolCall(
                            name=tc.get("function", {}).get("name", ""),
                            arguments="",
                            id=tc.get("id", ""),
                        )
                    func = tc.get("function", {})
                    if func.get("name"):
                        tool_calls_by_index[idx].name = func["name"]
                    if func.get("arguments"):
                        tool_calls_by_index[idx].arguments += func["arguments"]
                    if tc.get("id"):
                        tool_calls_by_index[idx].id = tc["id"]

        if not text_parts and not tool_calls_by_index:
            return None

        return Message(
            role="assistant",
            text="".join(text_parts) or None,
            tool_calls=[tool_calls_by_index[i] for i in sorted(tool_calls_by_index)],
            raw={},
        )

    def _parse_message(self, msg: dict) -> Message:
        role = msg.get("role", "unknown")
        content = msg.get("content")

        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            # OpenAI multimodal: [{"type": "text", "text": "..."}, ...]
            text_parts = [
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            text = "\n".join(text_parts) if text_parts else None
        else:
            text = None

        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            tool_calls.append(ToolCall(
                name=func.get("name", ""),
                arguments=func.get("arguments", ""),
                id=tc.get("id", ""),
            ))

        tool_result = None
        if role == "tool":
            tool_result = msg.get("content") if isinstance(msg.get("content"), str) else None

        return Message(
            role=role,
            text=text if role != "tool" else None,
            tool_calls=tool_calls,
            tool_result=tool_result,
            raw=msg,
        )

    def extract_session_hint(self, body: dict, headers: dict | None = None) -> str | None:
        # OpenAI has no standard session field; check common conventions
        metadata = body.get("metadata")
        if isinstance(metadata, dict):
            for key in ("session_id", "session"):
                val = metadata.get(key)
                if isinstance(val, str) and val:
                    return val
        user = body.get("user")
        if isinstance(user, str) and user:
            return user
        return None


# ---------------------------------------------------------------------------
# Anthropic parser
# ---------------------------------------------------------------------------


class AnthropicParser:
    """Parses Anthropic Messages API format."""

    def can_parse_request(self, body: dict) -> bool:
        # Anthropic requests may have a top-level "system" field (string or list),
        # or messages with content blocks containing "type": "tool_use"/"tool_result".
        if "system" in body:
            return True
        for msg in body.get("messages", []):
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") in (
                        "text", "tool_use", "tool_result", "image",
                    ):
                        return True
        return False

    def parse_request_messages(self, body: dict) -> list[Message]:
        result: list[Message] = []

        # System prompt
        system = body.get("system")
        if system:
            if isinstance(system, str):
                text = system
            elif isinstance(system, list):
                text = " ".join(
                    b.get("text", "") for b in system if b.get("type") == "text"
                )
            else:
                text = str(system)
            if text:
                result.append(Message(role="system", text=text, raw={"role": "system", "content": system}))

        for msg in body.get("messages", []):
            result.append(self._parse_message(msg))

        return result

    def can_parse_response(self, body: dict) -> bool:
        return body.get("role") == "assistant" and "content" in body and "choices" not in body

    def parse_response(self, body: dict) -> Message | None:
        return self._parse_message(body)

    def assemble_chunks(self, chunks: list[dict]) -> Message | None:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        current_tool: ToolCall | None = None

        for chunk in chunks:
            chunk_type = chunk.get("type", "")

            if chunk_type == "content_block_start":
                block = chunk.get("content_block", {})
                if block.get("type") == "tool_use":
                    current_tool = ToolCall(
                        name=block.get("name", ""),
                        id=block.get("id", ""),
                    )

            elif chunk_type == "content_block_delta":
                delta = chunk.get("delta", {})
                if delta.get("type") == "text_delta":
                    text_parts.append(delta.get("text", ""))
                elif delta.get("type") == "input_json_delta" and current_tool:
                    current_tool.arguments += delta.get("partial_json", "")

            elif chunk_type == "content_block_stop":
                if current_tool:
                    tool_calls.append(current_tool)
                    current_tool = None

        if not text_parts and not tool_calls:
            return None

        return Message(
            role="assistant",
            text="".join(text_parts) or None,
            tool_calls=tool_calls,
            raw={},
        )

    def _parse_message(self, msg: dict) -> Message:
        role = msg.get("role", "unknown")
        content = msg.get("content")

        if isinstance(content, str):
            return Message(role=role, text=content, raw=msg)

        if not isinstance(content, list):
            return Message(role=role, raw=msg)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        tool_results: list[str] = []

        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append(ToolCall(
                    name=block.get("name", ""),
                    arguments=json.dumps(block.get("input", {})),
                    id=block.get("id", ""),
                ))
            elif btype == "tool_result":
                tc = block.get("content", "")
                if isinstance(tc, list):
                    tc = " ".join(
                        b.get("text", "") for b in tc if b.get("type") == "text"
                    )
                tool_results.append(str(tc))

        text = "\n".join(text_parts) if text_parts else None
        tool_result = "\n".join(tool_results) if tool_results else None

        return Message(
            role=role,
            text=text,
            tool_calls=tool_calls,
            tool_result=tool_result,
            raw=msg,
        )

    def extract_session_hint(self, body: dict, headers: dict | None = None) -> str | None:
        # OpenCode sends session ID in x-session-affinity header
        if headers:
            sid = headers.get("x-session-affinity") or headers.get("X-Session-Affinity")
            if isinstance(sid, str) and sid:
                return sid
        # Anthropic clients (e.g. Claude Code) may put session info in metadata
        metadata = body.get("metadata")
        if isinstance(metadata, dict):
            for key in ("session_id", "session"):
                val = metadata.get(key)
                if isinstance(val, str) and val:
                    return val
            # Claude Code encodes session_id inside user_id as JSON
            user_id = metadata.get("user_id", "")
            if isinstance(user_id, str) and user_id.startswith("{"):
                try:
                    parsed = json.loads(user_id)
                    sid = parsed.get("session_id")
                    if isinstance(sid, str) and sid:
                        return sid
                except (json.JSONDecodeError, AttributeError):
                    pass
        return None


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

PARSERS: list[FormatParser] = [AnthropicParser(), OpenAIParser()]


def _find_request_parser(body: dict) -> FormatParser | None:
    for parser in PARSERS:
        if parser.can_parse_request(body):
            return parser
    return None


def _find_response_parser(body: dict) -> FormatParser | None:
    for parser in PARSERS:
        if parser.can_parse_response(body):
            return parser
    return None


def parse_record(record: dict) -> Conversation:
    """Convert a raw storage record into a canonical Conversation."""
    req_body = record.get("request_body") or {}
    resp_body = record.get("response_body") or {}
    chunks = record.get("response_chunks") or []

    # Parse request messages
    req_parser = _find_request_parser(req_body)
    messages = req_parser.parse_request_messages(req_body) if req_parser else []

    # Parse response
    resp_msg: Message | None = None
    if resp_body:
        resp_parser = _find_response_parser(resp_body)
        if resp_parser:
            resp_msg = resp_parser.parse_response(resp_body)

    # Fall back to chunk assembly if no response_body
    if resp_msg is None and chunks:
        for parser in PARSERS:
            resp_msg = parser.assemble_chunks(chunks)
            if resp_msg is not None:
                break

    if resp_msg is not None:
        messages.append(resp_msg)

    return Conversation(
        messages=messages,
        model=record.get("model") or "",
        timestamp=record.get("timestamp") or "",
        stream=bool(record.get("stream")),
        status_code=record.get("status_code"),
        duration_ms=record.get("duration_ms"),
        error=record.get("error"),
    )


def extract_session_hint(body: dict, headers: dict | None = None) -> str | None:
    """Try each parser to extract a session hint from a request body (and headers)."""
    for parser in PARSERS:
        if parser.can_parse_request(body):
            hint = parser.extract_session_hint(body, headers)
            if hint:
                return hint
    return None


def group_records(records: list[dict]) -> list[SessionGroup]:
    """Group records into sessions.

    Strategy:
    1. If a parser returns a session hint, group by that hint.
    2. Otherwise, each record becomes its own group (no guessing).

    Records must be sorted by timestamp DESC (newest first).
    Returns groups sorted by latest_timestamp DESC.
    """
    hinted: dict[str, list[dict]] = {}
    unhinted: list[dict] = []

    for record in records:
        body = record.get("request_body") or {}
        headers = record.get("request_headers") or {}
        hint = extract_session_hint(body, headers)
        if hint:
            hinted.setdefault(hint, []).append(record)
        else:
            unhinted.append(record)

    groups: list[SessionGroup] = []

    # Build groups from hinted records (sorted DESC, so first=newest, last=oldest)
    for hint, recs in hinted.items():
        models = dict.fromkeys(r.get("model") or "" for r in recs)
        groups.append(SessionGroup(
            session_id=hint,
            record_ids=[r["id"] for r in recs],
            model=", ".join(m for m in models if m) or "",
            earliest_timestamp=recs[-1].get("timestamp") or "",
            latest_timestamp=recs[0].get("timestamp") or "",
            request_count=len(recs),
        ))

    # Unhinted records: each is its own group
    for record in unhinted:
        model = record.get("model") or ""
        ts_str = record.get("timestamp") or ""
        groups.append(SessionGroup(
            session_id=record["id"],
            record_ids=[record["id"]],
            model=model,
            earliest_timestamp=ts_str,
            latest_timestamp=ts_str,
            request_count=1,
        ))

    groups.sort(key=lambda g: g.latest_timestamp, reverse=True)
    return groups


def assemble_openai_chunks(chunks: list[dict]) -> dict:
    """Assemble OpenAI streaming chunks into a chat.completion-like dict.

    Used by the proxy to store an assembled response_body alongside
    the raw chunks.  This is intentionally a raw-dict operation (not
    normalized) because the proxy stores wire-format data.
    """
    if not chunks:
        return {}

    first = chunks[0]
    last = chunks[-1]

    assembled_content = ""
    tool_calls_by_index: dict[int, dict] = {}
    finish_reason = None

    for chunk in chunks:
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})

            content = delta.get("content")
            if content:
                assembled_content += content

            if delta.get("tool_calls"):
                for tc in delta["tool_calls"]:
                    idx = tc.get("index", 0)
                    if idx not in tool_calls_by_index:
                        tool_calls_by_index[idx] = {
                            "id": tc.get("id", ""),
                            "type": tc.get("type", "function"),
                            "function": {"name": "", "arguments": ""},
                        }
                    if "function" in tc:
                        if tc["function"].get("name"):
                            tool_calls_by_index[idx]["function"]["name"] = tc["function"]["name"]
                        if tc["function"].get("arguments"):
                            tool_calls_by_index[idx]["function"]["arguments"] += tc["function"]["arguments"]
                    if tc.get("id"):
                        tool_calls_by_index[idx]["id"] = tc["id"]

            fr = choice.get("finish_reason")
            if fr:
                finish_reason = fr

    message: dict = {"role": "assistant", "content": assembled_content or None}
    if tool_calls_by_index:
        message["tool_calls"] = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)]

    result: dict = {
        "id": first.get("id", ""),
        "object": "chat.completion",
        "created": first.get("created", 0),
        "model": first.get("model", ""),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
    }

    usage = last.get("usage")
    if usage:
        result["usage"] = usage

    return result
