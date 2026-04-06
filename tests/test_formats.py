"""Tests for format detection, parsing, and dispatch."""

import json

from malcolm.formats import (
    AnthropicParser,
    OpenAIParser,
    assemble_openai_chunks,
    extract_session_hint,
    group_records,
    parse_record,
)


# ---------------------------------------------------------------------------
# OpenAI parser
# ---------------------------------------------------------------------------

class TestOpenAIParser:
    parser = OpenAIParser()

    def test_can_parse_request_simple(self):
        body = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        assert self.parser.can_parse_request(body) is True

    def test_can_parse_request_rejects_anthropic(self):
        body = {
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "ok"}]}],
        }
        assert self.parser.can_parse_request(body) is False

    def test_can_parse_request_rejects_empty(self):
        assert self.parser.can_parse_request({}) is False
        assert self.parser.can_parse_request({"messages": []}) is False

    def test_parse_request_messages(self):
        body = {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ]
        }
        msgs = self.parser.parse_request_messages(body)
        assert len(msgs) == 3
        assert msgs[0].role == "system"
        assert msgs[0].text == "You are helpful"
        assert msgs[1].role == "user"
        assert msgs[1].text == "hello"
        assert msgs[2].role == "assistant"
        assert msgs[2].text == "hi there"

    def test_parse_request_with_tool_calls(self):
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "London"}'}},
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            ]
        }
        msgs = self.parser.parse_request_messages(body)
        assert len(msgs) == 2
        assert msgs[0].tool_calls[0].name == "get_weather"
        assert msgs[0].text is None
        assert msgs[1].role == "tool"
        assert msgs[1].tool_result == "sunny"

    def test_parse_request_multimodal(self):
        body = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "What's this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ]},
            ]
        }
        msgs = self.parser.parse_request_messages(body)
        assert msgs[0].text == "What's this?"

    def test_can_parse_response(self):
        resp = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        assert self.parser.can_parse_response(resp) is True
        assert self.parser.can_parse_response({"role": "assistant", "content": []}) is False

    def test_parse_response(self):
        resp = {
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "hello!"}, "finish_reason": "stop"}
            ]
        }
        msg = self.parser.parse_response(resp)
        assert msg is not None
        assert msg.role == "assistant"
        assert msg.text == "hello!"

    def test_parse_response_with_tool_calls(self):
        resp = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                    ],
                },
            }]
        }
        msg = self.parser.parse_response(resp)
        assert msg is not None
        assert msg.text is None
        assert msg.tool_calls[0].name == "search"

    def test_assemble_chunks(self):
        chunks = [
            {"choices": [{"delta": {"role": "assistant", "content": "hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        msg = self.parser.assemble_chunks(chunks)
        assert msg is not None
        assert msg.text == "hello"
        assert msg.role == "assistant"

    def test_assemble_chunks_with_tool_calls(self):
        chunks = [
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "read", "arguments": ""}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"pa'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'th":"x"}'}}]}}]},
        ]
        msg = self.parser.assemble_chunks(chunks)
        assert msg is not None
        assert msg.tool_calls[0].name == "read"
        assert msg.tool_calls[0].arguments == '{"path":"x"}'

    def test_assemble_chunks_empty(self):
        assert self.parser.assemble_chunks([]) is None
        assert self.parser.assemble_chunks([{"choices": [{"delta": {}}]}]) is None


# ---------------------------------------------------------------------------
# Anthropic parser
# ---------------------------------------------------------------------------

class TestAnthropicParser:
    parser = AnthropicParser()

    def test_can_parse_request_with_system(self):
        body = {"model": "claude-sonnet-4-20250514", "system": "Be helpful", "messages": []}
        assert self.parser.can_parse_request(body) is True

    def test_can_parse_request_with_content_blocks(self):
        body = {
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        }
        assert self.parser.can_parse_request(body) is True

    def test_can_parse_request_rejects_plain(self):
        assert self.parser.can_parse_request({}) is False
        assert self.parser.can_parse_request({"messages": [{"role": "user", "content": "hi"}]}) is False

    def test_parse_request_with_system_string(self):
        body = {
            "system": "You are a poet",
            "messages": [{"role": "user", "content": "write a haiku"}],
        }
        msgs = self.parser.parse_request_messages(body)
        assert msgs[0].role == "system"
        assert msgs[0].text == "You are a poet"
        assert msgs[1].role == "user"
        assert msgs[1].text == "write a haiku"

    def test_parse_request_with_system_blocks(self):
        body = {
            "system": [{"type": "text", "text": "Be concise"}, {"type": "text", "text": "Be clear"}],
            "messages": [],
        }
        msgs = self.parser.parse_request_messages(body)
        assert msgs[0].text == "Be concise Be clear"

    def test_parse_request_with_tool_use(self):
        body = {
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "id": "tu_1", "name": "search", "input": {"q": "test"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1", "content": "found it"},
                ]},
            ]
        }
        msgs = self.parser.parse_request_messages(body)
        assert msgs[0].text == "Let me check."
        assert msgs[0].tool_calls[0].name == "search"
        assert msgs[0].tool_calls[0].arguments == json.dumps({"q": "test"})
        assert msgs[1].tool_result == "found it"

    def test_can_parse_response(self):
        resp = {"role": "assistant", "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn"}
        assert self.parser.can_parse_response(resp) is True

    def test_can_parse_response_rejects_openai(self):
        resp = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        assert self.parser.can_parse_response(resp) is False

    def test_parse_response(self):
        resp = {
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
        }
        msg = self.parser.parse_response(resp)
        assert msg is not None
        assert msg.role == "assistant"
        assert msg.text == "Hello!"

    def test_parse_response_with_tool_use(self):
        resp = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Searching..."},
                {"type": "tool_use", "id": "tu_1", "name": "web_search", "input": {"query": "test"}},
            ],
        }
        msg = self.parser.parse_response(resp)
        assert msg is not None
        assert msg.text == "Searching..."
        assert msg.tool_calls[0].name == "web_search"

    def test_assemble_chunks(self):
        chunks = [
            {"type": "message_start", "message": {"id": "msg_1", "role": "assistant", "content": []}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hel"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        ]
        msg = self.parser.assemble_chunks(chunks)
        assert msg is not None
        assert msg.text == "hello"

    def test_assemble_chunks_with_tool_use(self):
        chunks = [
            {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "tu_1", "name": "search"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"q":'}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '"test"}'}},
            {"type": "content_block_stop", "index": 0},
        ]
        msg = self.parser.assemble_chunks(chunks)
        assert msg is not None
        assert msg.tool_calls[0].name == "search"
        assert msg.tool_calls[0].arguments == '{"q":"test"}'

    def test_assemble_chunks_empty(self):
        assert self.parser.assemble_chunks([]) is None


# ---------------------------------------------------------------------------
# Dispatch (parse_record)
# ---------------------------------------------------------------------------

class TestParseRecord:
    def test_openai_request_and_response(self):
        record = {
            "model": "gpt-4",
            "timestamp": "2026-01-01T00:00:00",
            "stream": 0,
            "status_code": 200,
            "request_body": {
                "messages": [{"role": "user", "content": "hello"}],
            },
            "response_body": {
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            },
        }
        conv = parse_record(record)
        assert conv.model == "gpt-4"
        assert len(conv.messages) == 2
        assert conv.messages[0].role == "user"
        assert conv.messages[0].text == "hello"
        assert conv.messages[1].role == "assistant"
        assert conv.messages[1].text == "hi"

    def test_anthropic_request_and_response(self):
        record = {
            "model": "claude-sonnet-4-20250514",
            "timestamp": "2026-01-01T00:00:00",
            "stream": 0,
            "request_body": {
                "system": "Be helpful",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            },
            "response_body": {
                "role": "assistant",
                "content": [{"type": "text", "text": "hi"}],
            },
        }
        conv = parse_record(record)
        assert len(conv.messages) == 3  # system + user + assistant
        assert conv.messages[0].role == "system"
        assert conv.messages[1].text == "hello"
        assert conv.messages[2].text == "hi"

    def test_streaming_fallback_to_chunks(self):
        record = {
            "model": "gpt-4",
            "timestamp": "2026-01-01T00:00:00",
            "stream": 1,
            "request_body": {
                "messages": [{"role": "user", "content": "hi"}],
            },
            "response_body": None,
            "response_chunks": [
                {"choices": [{"delta": {"role": "assistant", "content": "hel"}}]},
                {"choices": [{"delta": {"content": "lo"}}]},
            ],
        }
        conv = parse_record(record)
        assert len(conv.messages) == 2
        assert conv.messages[1].text == "hello"

    def test_empty_record(self):
        record = {"model": "", "timestamp": "", "request_body": {}, "response_body": {}}
        conv = parse_record(record)
        assert conv.messages == []

    def test_unknown_format_returns_empty_messages(self):
        record = {
            "model": "custom",
            "timestamp": "2026-01-01T00:00:00",
            "request_body": {"prompt": "hello"},
            "response_body": {"text": "hi"},
        }
        conv = parse_record(record)
        assert conv.messages == []
        assert conv.model == "custom"


# ---------------------------------------------------------------------------
# assemble_openai_chunks (used by proxy)
# ---------------------------------------------------------------------------

class TestAssembleOpenAIChunks:
    def test_basic_assembly(self):
        chunks = [
            {"id": "chatcmpl-1", "created": 1700000000, "model": "gpt-4",
             "choices": [{"delta": {"role": "assistant", "content": "hel"}}]},
            {"id": "chatcmpl-1", "created": 1700000000, "model": "gpt-4",
             "choices": [{"delta": {"content": "lo"}}]},
            {"id": "chatcmpl-1", "created": 1700000000, "model": "gpt-4",
             "choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        result = assemble_openai_chunks(chunks)
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "hello"
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_empty_chunks(self):
        assert assemble_openai_chunks([]) == {}

    def test_with_tool_calls(self):
        chunks = [
            {"id": "c1", "created": 0, "model": "gpt-4",
             "choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "read", "arguments": ""}}]}}]},
            {"id": "c1", "created": 0, "model": "gpt-4",
             "choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"x":1}'}}]}}]},
        ]
        result = assemble_openai_chunks(chunks)
        msg = result["choices"][0]["message"]
        assert msg["tool_calls"][0]["function"]["name"] == "read"
        assert msg["tool_calls"][0]["function"]["arguments"] == '{"x":1}'


# ---------------------------------------------------------------------------
# Session hints
# ---------------------------------------------------------------------------

class TestSessionHints:
    def test_openai_metadata_session_id(self):
        body = {"messages": [{"role": "user", "content": "hi"}], "metadata": {"session_id": "sess-1"}}
        assert extract_session_hint(body) == "sess-1"

    def test_openai_user_field(self):
        body = {"messages": [{"role": "user", "content": "hi"}], "user": "user-123"}
        assert extract_session_hint(body) == "user-123"

    def test_openai_no_hint(self):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        assert extract_session_hint(body) is None

    def test_anthropic_metadata_session_id(self):
        body = {"system": "Be helpful", "messages": [], "metadata": {"session_id": "sess-a"}}
        assert extract_session_hint(body) == "sess-a"

    def test_anthropic_claude_code_user_id(self):
        body = {
            "system": "Be helpful",
            "messages": [],
            "metadata": {"user_id": '{"session_id": "cc-sess-1"}'},
        }
        assert extract_session_hint(body) == "cc-sess-1"

    def test_anthropic_no_hint(self):
        body = {"system": "Be helpful", "messages": []}
        assert extract_session_hint(body) is None


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

class TestGroupRecords:
    def test_group_by_session_hint(self):
        records = [
            {"id": "r1", "timestamp": "2026-01-01T00:10:00", "model": "gpt-4",
             "request_body": {"messages": [{"role": "user", "content": "hi"}], "metadata": {"session_id": "s1"}}},
            {"id": "r2", "timestamp": "2026-01-01T00:00:00", "model": "gpt-4",
             "request_body": {"messages": [{"role": "user", "content": "hello"}], "metadata": {"session_id": "s1"}}},
        ]
        groups = group_records(records)
        assert len(groups) == 1
        assert groups[0].session_id == "s1"
        assert groups[0].record_ids == ["r1", "r2"]
        assert groups[0].request_count == 2

    def test_unhinted_records_not_grouped(self):
        """Records without session hints are never grouped together."""
        records = [
            {"id": "r1", "timestamp": "2026-01-01T00:05:00", "model": "gpt-4",
             "request_body": {"messages": [{"role": "user", "content": "hi"}]}},
            {"id": "r2", "timestamp": "2026-01-01T00:00:00", "model": "gpt-4",
             "request_body": {"messages": [{"role": "user", "content": "hello"}]}},
        ]
        groups = group_records(records)
        assert len(groups) == 2
        assert all(g.request_count == 1 for g in groups)

    def test_empty_records(self):
        assert group_records([]) == []

    def test_mixed_hinted_and_unhinted(self):
        records = [
            {"id": "r1", "timestamp": "2026-01-01T00:10:00", "model": "gpt-4",
             "request_body": {"messages": [{"role": "user", "content": "hi"}], "metadata": {"session_id": "s1"}}},
            {"id": "r2", "timestamp": "2026-01-01T00:05:00", "model": "gpt-4",
             "request_body": {"messages": [{"role": "user", "content": "hello"}]}},
            {"id": "r3", "timestamp": "2026-01-01T00:00:00", "model": "gpt-4",
             "request_body": {"messages": [{"role": "user", "content": "hey"}], "metadata": {"session_id": "s1"}}},
        ]
        groups = group_records(records)
        # r1 and r3 share hint "s1" → one group
        # r2 has no hint → its own group
        assert len(groups) == 2
        hinted = [g for g in groups if g.session_id == "s1"]
        assert len(hinted) == 1
        assert hinted[0].record_ids == ["r1", "r3"]
