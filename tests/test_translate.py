import json

import pytest

from malcolm.transforms.translation.engine import (
    anthropic_request_to_openai,
    anthropic_response_to_openai,
    anthropic_stream_to_openai_lines,
    openai_request_to_anthropic,
    openai_response_to_anthropic,
    openai_stream_to_anthropic_events,
    rewrite_path,
)


# ===================================================================
# Anthropic → OpenAI: request translation
# ===================================================================


class TestAnthropicRequestToOpenAI:
    def test_basic_text_message(self):
        body = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = anthropic_request_to_openai(body)

        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["max_tokens"] == 1024
        assert len(result["messages"]) == 1
        assert result["messages"][0] == {"role": "user", "content": "Hello"}

    def test_system_prompt_string(self):
        body = {
            "model": "test",
            "max_tokens": 100,
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = anthropic_request_to_openai(body)

        assert result["messages"][0] == {"role": "system", "content": "You are helpful."}
        assert result["messages"][1] == {"role": "user", "content": "Hi"}
        assert "system" not in result

    def test_system_prompt_blocks(self):
        body = {
            "model": "test",
            "max_tokens": 100,
            "system": [{"type": "text", "text": "Be concise."}, {"type": "text", "text": "Be helpful."}],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = anthropic_request_to_openai(body)

        assert result["messages"][0]["role"] == "system"
        assert "Be concise." in result["messages"][0]["content"]
        assert "Be helpful." in result["messages"][0]["content"]

    def test_tool_use_in_assistant_message(self):
        body = {
            "model": "test",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check."},
                        {
                            "type": "tool_use",
                            "id": "call_123",
                            "name": "get_weather",
                            "input": {"city": "NYC"},
                        },
                    ],
                },
            ],
        }
        result = anthropic_request_to_openai(body)

        assistant_msg = result["messages"][1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] == "Let me check."
        assert len(assistant_msg["tool_calls"]) == 1
        tc = assistant_msg["tool_calls"][0]
        assert tc["id"] == "call_123"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "NYC"}

    def test_tool_result_in_user_message(self):
        body = {
            "model": "test",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "call_123", "content": "Sunny, 72°F"},
                    ],
                },
            ],
        }
        result = anthropic_request_to_openai(body)

        assert result["messages"][0]["role"] == "tool"
        assert result["messages"][0]["tool_call_id"] == "call_123"
        assert result["messages"][0]["content"] == "Sunny, 72°F"

    def test_tool_definitions(self):
        body = {
            "model": "test",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
                }
            ],
        }
        result = anthropic_request_to_openai(body)

        tool = result["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"
        assert tool["function"]["description"] == "Get weather for a city"
        assert tool["function"]["parameters"]["type"] == "object"

    def test_image_base64(self):
        body = {
            "model": "test",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": "abc123"},
                        },
                        {"type": "text", "text": "What's this?"},
                    ],
                }
            ],
        }
        result = anthropic_request_to_openai(body)

        user_msg = result["messages"][0]
        assert user_msg["content"][0]["type"] == "image_url"
        assert user_msg["content"][0]["image_url"]["url"] == "data:image/png;base64,abc123"
        assert user_msg["content"][1]["type"] == "text"

    def test_stop_sequences(self):
        body = {
            "model": "test",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "stop_sequences": ["\n\n"],
        }
        result = anthropic_request_to_openai(body)

        assert result["stop"] == ["\n\n"]
        assert "stop_sequences" not in result

    def test_preserves_shared_fields(self):
        body = {
            "model": "test",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": True,
        }
        result = anthropic_request_to_openai(body)

        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["stream"] is True


# ===================================================================
# Anthropic → OpenAI: response translation (OpenAI resp → Anthropic resp)
# ===================================================================


class TestOpenAIResponseToAnthropic:
    def test_basic_text(self):
        openai_resp = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4.1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = openai_response_to_anthropic(openai_resp, "gpt-4.1")

        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["content"] == [{"type": "text", "text": "Hello!"}]
        assert result["stop_reason"] == "end_turn"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5

    def test_tool_calls(self):
        openai_resp = {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tc_1",
                                "type": "function",
                                "function": {"name": "search", "arguments": '{"q": "test"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        result = openai_response_to_anthropic(openai_resp, "gpt-4.1")

        assert result["stop_reason"] == "tool_use"
        assert len(result["content"]) == 1
        tc = result["content"][0]
        assert tc["type"] == "tool_use"
        assert tc["id"] == "tc_1"
        assert tc["name"] == "search"
        assert tc["input"] == {"q": "test"}


# ===================================================================
# OpenAI → Anthropic: request translation
# ===================================================================


class TestOpenAIRequestToAnthropic:
    def test_basic_text_message(self):
        body = {
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = openai_request_to_anthropic(body)

        assert result["model"] == "gpt-4.1"
        assert result["max_tokens"] == 4096  # default
        assert len(result["messages"]) == 1
        assert result["messages"][0] == {"role": "user", "content": "Hello"}

    def test_system_message_extraction(self):
        body = {
            "model": "test",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
        }
        result = openai_request_to_anthropic(body)

        assert result["system"] == "You are helpful."
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_tool_calls_in_assistant(self):
        body = {
            "model": "test",
            "messages": [
                {"role": "user", "content": "Search for cats"},
                {
                    "role": "assistant",
                    "content": "Let me search.",
                    "tool_calls": [
                        {
                            "id": "tc_1",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"q": "cats"}'},
                        }
                    ],
                },
            ],
        }
        result = openai_request_to_anthropic(body)

        assistant_msg = result["messages"][1]
        assert assistant_msg["role"] == "assistant"
        assert len(assistant_msg["content"]) == 2
        assert assistant_msg["content"][0] == {"type": "text", "text": "Let me search."}
        assert assistant_msg["content"][1]["type"] == "tool_use"
        assert assistant_msg["content"][1]["name"] == "search"
        assert assistant_msg["content"][1]["input"] == {"q": "cats"}

    def test_tool_messages_become_tool_results(self):
        body = {
            "model": "test",
            "messages": [
                {"role": "tool", "tool_call_id": "tc_1", "content": "Found 10 results"},
            ],
        }
        result = openai_request_to_anthropic(body)

        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"][0]["type"] == "tool_result"
        assert result["messages"][0]["content"][0]["tool_use_id"] == "tc_1"

    def test_consecutive_tool_messages_merged(self):
        body = {
            "model": "test",
            "messages": [
                {"role": "tool", "tool_call_id": "tc_1", "content": "Result 1"},
                {"role": "tool", "tool_call_id": "tc_2", "content": "Result 2"},
            ],
        }
        result = openai_request_to_anthropic(body)

        assert len(result["messages"]) == 1
        assert len(result["messages"][0]["content"]) == 2

    def test_tool_definitions(self):
        body = {
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search the web",
                        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                    },
                }
            ],
        }
        result = openai_request_to_anthropic(body)

        tool = result["tools"][0]
        assert tool["name"] == "search"
        assert tool["description"] == "Search the web"
        assert tool["input_schema"]["type"] == "object"

    def test_stop_to_stop_sequences(self):
        body = {
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": ["\n"],
        }
        result = openai_request_to_anthropic(body)

        assert result["stop_sequences"] == ["\n"]

    def test_stop_string_to_list(self):
        body = {
            "model": "test",
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": "\n",
        }
        result = openai_request_to_anthropic(body)

        assert result["stop_sequences"] == ["\n"]

    def test_image_data_uri(self):
        body = {
            "model": "test",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
                        {"type": "text", "text": "What's this?"},
                    ],
                }
            ],
        }
        result = openai_request_to_anthropic(body)

        content = result["messages"][0]["content"]
        assert content[0]["type"] == "image"
        assert content[0]["source"]["type"] == "base64"
        assert content[0]["source"]["media_type"] == "image/png"
        assert content[0]["source"]["data"] == "abc123"


# ===================================================================
# OpenAI → Anthropic: response translation (Anthropic resp → OpenAI resp)
# ===================================================================


class TestAnthropicResponseToOpenAI:
    def test_basic_text(self):
        anthropic_resp = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = anthropic_response_to_openai(anthropic_resp)

        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "Hello!"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15

    def test_tool_use(self):
        anthropic_resp = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tc_1", "name": "search", "input": {"q": "test"}},
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
        result = anthropic_response_to_openai(anthropic_resp)

        assert result["choices"][0]["finish_reason"] == "tool_calls"
        tc = result["choices"][0]["message"]["tool_calls"][0]
        assert tc["id"] == "tc_1"
        assert tc["function"]["name"] == "search"
        assert json.loads(tc["function"]["arguments"]) == {"q": "test"}


# ===================================================================
# Streaming: OpenAI SSE → Anthropic SSE
# ===================================================================


class TestOpenAIStreamToAnthropicEvents:
    def test_text_stream(self):
        state: dict = {}
        all_events: list[str] = []

        lines = [
            'data: {"id":"c1","model":"gpt-4.1","choices":[{"delta":{"role":"assistant","content":""},"index":0}]}',
            'data: {"id":"c1","model":"gpt-4.1","choices":[{"delta":{"content":"Hello"},"index":0}]}',
            'data: {"id":"c1","model":"gpt-4.1","choices":[{"delta":{"content":" world"},"index":0}]}',
            'data: {"id":"c1","model":"gpt-4.1","choices":[{"delta":{},"index":0,"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]

        for line in lines:
            all_events.extend(openai_stream_to_anthropic_events(line, state))

        event_types = [e.split("\n")[0].replace("event: ", "") for e in all_events]

        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types
        assert "message_delta" in event_types
        assert "message_stop" in event_types

    def test_tool_call_stream(self):
        state: dict = {}
        all_events: list[str] = []

        lines = [
            'data: {"id":"c1","model":"gpt-4.1","choices":[{"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"tc1","type":"function","function":{"name":"search","arguments":""}}]},"index":0}]}',
            'data: {"id":"c1","model":"gpt-4.1","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"q\\":"}}]},"index":0}]}',
            'data: {"id":"c1","model":"gpt-4.1","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"test\\"}"}}]},"index":0}]}',
            'data: {"id":"c1","model":"gpt-4.1","choices":[{"delta":{},"index":0,"finish_reason":"tool_calls"}]}',
            "data: [DONE]",
        ]

        for line in lines:
            all_events.extend(openai_stream_to_anthropic_events(line, state))

        event_types = [e.split("\n")[0].replace("event: ", "") for e in all_events]

        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert state.get("stop_reason") == "tool_use"


# ===================================================================
# Streaming: Anthropic SSE → OpenAI SSE
# ===================================================================


class TestAnthropicStreamToOpenAILines:
    def test_text_stream(self):
        state: dict = {}
        all_lines: list[str] = []

        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-20250514","stop_reason":null,"usage":{"input_tokens":10,"output_tokens":0}}}',
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}',
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}',
            "event: message_stop",
            'data: {"type":"message_stop"}',
        ]

        for line in sse_lines:
            all_lines.extend(anthropic_stream_to_openai_lines(line, state))

        # Parse emitted chunks
        chunks = []
        done = False
        for ln in all_lines:
            ln = ln.strip()
            if ln == "data: [DONE]":
                done = True
            elif ln.startswith("data: "):
                chunks.append(json.loads(ln[6:]))

        assert done
        assert len(chunks) >= 3  # role, 2x content, finish

        # First chunk has role
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"

        # Content chunks
        content_parts = [
            c["choices"][0]["delta"].get("content", "")
            for c in chunks
            if c["choices"][0]["delta"].get("content")
        ]
        assert "Hello" in content_parts
        assert " world" in content_parts

        # Last data chunk has finish_reason
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    def test_tool_use_stream(self):
        state: dict = {}
        all_lines: list[str] = []

        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-20250514","stop_reason":null,"usage":{"input_tokens":10,"output_tokens":0}}}',
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tc_1","name":"search","input":{}}}',
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"q\\": \\"test\\"}"}}',
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":3}}',
            "event: message_stop",
            'data: {"type":"message_stop"}',
        ]

        for line in sse_lines:
            all_lines.extend(anthropic_stream_to_openai_lines(line, state))

        chunks = []
        for ln in all_lines:
            ln = ln.strip()
            if ln.startswith("data: ") and ln != "data: [DONE]":
                chunks.append(json.loads(ln[6:]))

        # Should have tool_calls in deltas
        tool_chunks = [
            c for c in chunks if c["choices"][0]["delta"].get("tool_calls")
        ]
        assert len(tool_chunks) >= 1

        # Finish reason should be tool_calls
        finish_chunks = [c for c in chunks if c["choices"][0].get("finish_reason")]
        assert finish_chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


# ===================================================================
# Path rewriting
# ===================================================================


class TestRewritePath:
    def test_anthropic_to_openai(self):
        assert rewrite_path("/v1/messages", "anthropic_to_openai") == "/v1/chat/completions"

    def test_anthropic_to_openai_with_query(self):
        assert rewrite_path("/v1/messages?beta=true", "anthropic_to_openai") == "/v1/chat/completions"

    def test_openai_to_anthropic(self):
        assert rewrite_path("/v1/chat/completions", "openai_to_anthropic") == "/v1/messages"

    def test_no_translation(self):
        assert rewrite_path("/v1/messages", "") == "/v1/messages"
        assert rewrite_path("/v1/chat/completions", "") == "/v1/chat/completions"

    def test_unrelated_path_unchanged(self):
        assert rewrite_path("/v1/models", "anthropic_to_openai") == "/v1/models"
