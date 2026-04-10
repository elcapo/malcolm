from malcolm.transforms._base import Annotation
from malcolm.transforms.llm_annotator import LLMAnnotatorTransform, create


class TestLLMAnnotatorPassthrough:
    def test_transform_request_is_passthrough(self):
        t = LLMAnnotatorTransform()
        body = {"model": "gpt-4", "messages": []}
        assert t.transform_request(body) is body

    def test_transform_response_is_passthrough(self):
        t = LLMAnnotatorTransform()
        body = {"choices": []}
        assert t.transform_response(body) is body

    def test_transform_stream_line_is_passthrough(self):
        t = LLMAnnotatorTransform()
        assert t.transform_stream_line("data: {}", {}) == ["data: {}"]

    def test_rewrite_path_is_passthrough(self):
        t = LLMAnnotatorTransform()
        assert t.rewrite_path("/v1/messages") == "/v1/messages"


class TestAnnotateRequest:
    def test_extracts_model(self):
        t = LLMAnnotatorTransform()
        anns = t.annotate_request(
            request_body={"model": "gpt-4o", "messages": []},
        )
        models = [a for a in anns if a.key == "model"]
        assert len(models) == 1
        assert models[0].value == "gpt-4o"
        assert models[0].display == "badge"
        assert models[0].category == "metadata"

    def test_extracts_session_id(self):
        t = LLMAnnotatorTransform()
        anns = t.annotate_request(
            request_body={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "metadata": {"session_id": "sess-123"},
            },
        )
        sessions = [a for a in anns if a.key == "session_id"]
        assert len(sessions) == 1
        assert sessions[0].value == "sess-123"

    def test_extracts_system_prompt(self):
        t = LLMAnnotatorTransform()
        anns = t.annotate_request(
            request_body={
                "model": "claude-opus-4-6",
                "system": "You are helpful",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                ],
            },
        )
        system = [a for a in anns if a.key == "system_prompt"]
        assert len(system) == 1
        assert system[0].value == "You are helpful"
        assert system[0].display == "text"

    def test_extracts_user_messages(self):
        t = LLMAnnotatorTransform()
        anns = t.annotate_request(
            request_body={
                "model": "gpt-4o",
                "messages": [
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi!"},
                    {"role": "user", "content": "How are you?"},
                ],
            },
        )
        users = [a for a in anns if a.key.startswith("user_message.")]
        assert len(users) == 2
        assert users[0].key == "user_message.0"
        assert users[0].value == "Hello!"
        assert users[1].key == "user_message.1"
        assert users[1].value == "How are you?"

    def test_no_model_no_annotation(self):
        t = LLMAnnotatorTransform()
        anns = t.annotate_request(request_body={})
        models = [a for a in anns if a.key == "model"]
        assert len(models) == 0

    def test_stream_badge(self):
        t = LLMAnnotatorTransform()
        anns = t.annotate_request(
            request_body={"model": "gpt-4o", "messages": [], "stream": True},
        )
        streams = [a for a in anns if a.key == "stream"]
        assert len(streams) == 1
        assert streams[0].display == "badge"


class TestAnnotateResponse:
    def test_extracts_tool_calls(self):
        t = LLMAnnotatorTransform()
        anns = t.annotate_response(
            response_body={
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "function": {"name": "read_file", "arguments": "{}"},
                            "type": "function",
                            "id": "tc_1",
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
            },
        )
        tools = [a for a in anns if a.key.startswith("tool_call.")]
        assert len(tools) == 1
        assert tools[0].value == "read_file"

    def test_extracts_assistant_message(self):
        t = LLMAnnotatorTransform()
        anns = t.annotate_response(
            response_body={
                "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
            },
        )
        msgs = [a for a in anns if a.key.startswith("assistant_message.")]
        assert len(msgs) == 1
        assert msgs[0].value == "Hello!"
        assert msgs[0].display == "text"

    def test_extracts_usage(self):
        t = LLMAnnotatorTransform()
        anns = t.annotate_response(
            response_body={
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                },
            },
        )
        usage = {a.key: a.value for a in anns if a.category == "usage"}
        assert usage["input_tokens"] == "10"
        assert usage["output_tokens"] == "5"

    def test_empty_response_body(self):
        t = LLMAnnotatorTransform()
        anns = t.annotate_response(response_body=None)
        assert anns == []

    def test_does_not_include_request_fields(self):
        """annotate_response should not extract model or session_id."""
        t = LLMAnnotatorTransform()
        anns = t.annotate_response(
            response_body={
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            },
        )
        assert not any(a.key == "model" for a in anns)
        assert not any(a.key == "session_id" for a in anns)


class TestLLMAnnotatorFactory:
    def test_create_returns_instance(self):
        t = create({})
        assert isinstance(t, LLMAnnotatorTransform)
        assert t.name == "llm_annotator"

    def test_has_annotate_methods(self):
        t = create({})
        assert hasattr(t, "annotate_request")
        assert hasattr(t, "annotate_response")
