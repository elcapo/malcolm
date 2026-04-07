import pytest

from malcolm.config import Settings
from malcolm.ghostkey import reset_session
from malcolm.transforms import (
    GhostKeyTransform,
    TranslationTransform,
    build_pipeline,
)


@pytest.fixture(autouse=True)
def _clean_ghostkey():
    reset_session()
    yield
    reset_session()


# ── GhostKeyTransform ───────────────────────────────────────────────────


class TestGhostKeyTransform:
    def test_obfuscates_secrets_in_request(self):
        t = GhostKeyTransform()
        body = {
            "messages": [
                {"role": "user", "content": "my key is sk-ant-ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
            ]
        }
        result = t.transform_request(body)
        assert "sk-ant-ABCDEFGHIJKLMNOPQRSTUVWXYZ" not in str(result)
        # Prefix up to first separator is preserved
        assert result["messages"][0]["content"].startswith("my key is sk-")

    def test_restores_secrets_in_response(self):
        t = GhostKeyTransform()
        body = {
            "messages": [
                {"role": "user", "content": "my key is sk-ant-ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
            ]
        }
        obfuscated = t.transform_request(body)
        # Simulate backend echoing the obfuscated key
        response = {"content": obfuscated["messages"][0]["content"]}
        restored = t.transform_response(response)
        assert "sk-ant-ABCDEFGHIJKLMNOPQRSTUVWXYZ" in restored["content"]

    def test_roundtrip(self):
        t = GhostKeyTransform()
        secret = "sk-ant-ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        body = {"messages": [{"role": "user", "content": f"key={secret}"}]}

        obfuscated = t.transform_request(body)
        assert secret not in str(obfuscated)

        restored = t.transform_response(obfuscated)
        assert secret in str(restored)

    def test_no_secrets_passthrough(self):
        t = GhostKeyTransform()
        body = {"messages": [{"role": "user", "content": "hello"}]}
        result = t.transform_request(body)
        assert result == body

    def test_stream_line_restore(self):
        t = GhostKeyTransform()
        # Register a secret first
        t.transform_request({"messages": [{"role": "user", "content": "sk-ant-ABCDEFGHIJKLMNOPQRSTUVWXYZ"}]})
        obfuscated = t.transform_request({"messages": [{"role": "user", "content": "sk-ant-ABCDEFGHIJKLMNOPQRSTUVWXYZ"}]})
        fake_key = obfuscated["messages"][0]["content"]

        lines = t.transform_stream_line(f'data: {{"content": "{fake_key}"}}', {})
        assert len(lines) == 1
        assert "sk-ant-ABCDEFGHIJKLMNOPQRSTUVWXYZ" in lines[0]

    def test_rewrite_path_noop(self):
        t = GhostKeyTransform()
        assert t.rewrite_path("/v1/messages") == "/v1/messages"


# ── TranslationTransform ────────────────────────────────────────────────


class TestTranslationTransform:
    def test_anthropic_to_openai_request(self):
        t = TranslationTransform("anthropic_to_openai")
        body = {
            "model": "claude-3-opus-20240229",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hello"}],
        }
        result = t.transform_request(body)
        assert "messages" in result
        assert result["model"] == "claude-3-opus-20240229"
        assert result.get("max_tokens") or result.get("max_completion_tokens")

    def test_anthropic_to_openai_response(self):
        t = TranslationTransform("anthropic_to_openai")
        openai_response = {
            "id": "chatcmpl-1",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "model": "gpt-4",
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        }
        result = t.transform_response(openai_response, model="claude-3-opus-20240229")
        assert result["type"] == "message"
        assert result["role"] == "assistant"

    def test_openai_to_anthropic_request(self):
        t = TranslationTransform("openai_to_anthropic")
        body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
        }
        result = t.transform_request(body)
        assert result["model"] == "gpt-4"
        assert "max_tokens" in result

    def test_openai_to_anthropic_response(self):
        t = TranslationTransform("openai_to_anthropic")
        anthropic_response = {
            "id": "msg-1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "model": "claude-3-opus-20240229",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 1},
        }
        result = t.transform_response(anthropic_response)
        assert "choices" in result
        assert result["choices"][0]["message"]["content"] == "hi"

    def test_rewrite_path_anthropic_to_openai(self):
        t = TranslationTransform("anthropic_to_openai")
        assert "/chat/completions" in t.rewrite_path("/v1/messages")

    def test_rewrite_path_openai_to_anthropic(self):
        t = TranslationTransform("openai_to_anthropic")
        assert "/messages" in t.rewrite_path("/v1/chat/completions")

    def test_no_direction_passthrough(self):
        t = TranslationTransform("")
        body = {"model": "gpt-4", "messages": []}
        assert t.transform_request(body) == body
        assert t.transform_response(body) == body

    def test_stream_line_anthropic_to_openai(self):
        t = TranslationTransform("anthropic_to_openai")
        state = {}
        # OpenAI SSE line → should produce Anthropic SSE events
        line = 'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}'
        result = t.transform_stream_line(line, state)
        assert isinstance(result, list)

    def test_stream_line_openai_to_anthropic(self):
        t = TranslationTransform("openai_to_anthropic")
        state = {}
        line = 'event: message_start'
        result = t.transform_stream_line(line, state)
        assert isinstance(result, list)


# ── build_pipeline ──────────────────────────────────────────────────────


class TestBuildPipeline:
    def test_empty_pipeline(self, monkeypatch):
        monkeypatch.setenv("MALCOLM_TARGET_URL", "http://test")
        monkeypatch.setenv("MALCOLM_GHOSTKEY_ENABLED", "false")
        monkeypatch.delenv("MALCOLM_TRANSLATE", raising=False)
        settings = Settings(_env_file=None)
        pipeline = build_pipeline(settings)
        assert pipeline == []

    def test_ghostkey_only(self, monkeypatch):
        monkeypatch.setenv("MALCOLM_TARGET_URL", "http://test")
        monkeypatch.setenv("MALCOLM_GHOSTKEY_ENABLED", "true")
        monkeypatch.delenv("MALCOLM_TRANSLATE", raising=False)
        settings = Settings(_env_file=None)
        pipeline = build_pipeline(settings)
        assert len(pipeline) == 1
        assert pipeline[0].name == "ghostkey"

    def test_translation_only(self, monkeypatch):
        monkeypatch.setenv("MALCOLM_TARGET_URL", "http://test")
        monkeypatch.setenv("MALCOLM_GHOSTKEY_ENABLED", "false")
        monkeypatch.setenv("MALCOLM_TRANSLATE", "anthropic_to_openai")
        settings = Settings(_env_file=None)
        pipeline = build_pipeline(settings)
        assert len(pipeline) == 1
        assert pipeline[0].name == "translation"

    def test_both_ghostkey_first(self, monkeypatch):
        monkeypatch.setenv("MALCOLM_TARGET_URL", "http://test")
        monkeypatch.setenv("MALCOLM_GHOSTKEY_ENABLED", "true")
        monkeypatch.setenv("MALCOLM_TRANSLATE", "openai_to_anthropic")
        settings = Settings(_env_file=None)
        pipeline = build_pipeline(settings)
        assert len(pipeline) == 2
        assert pipeline[0].name == "ghostkey"
        assert pipeline[1].name == "translation"
