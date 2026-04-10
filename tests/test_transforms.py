import pytest
import yaml

import malcolm.transforms as transforms_module
from malcolm.transforms.ghostkey.engine import reset_session
from malcolm.transforms import (
    REGISTRY,
    GhostKeyTransform,
    LLMAnnotatorTransform,
    TranslationTransform,
    _discover_entry_points,
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
        assert result["messages"][0]["content"].startswith("my key is sk-")

    def test_restores_secrets_in_response(self):
        t = GhostKeyTransform()
        body = {
            "messages": [
                {"role": "user", "content": "my key is sk-ant-ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
            ]
        }
        obfuscated = t.transform_request(body)
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

    def test_anthropic_to_openai_error_passthrough(self):
        """OpenAI error bodies are forwarded unchanged instead of being translated."""
        t = TranslationTransform("anthropic_to_openai")
        err = {"error": {"message": "invalid key", "type": "authentication_error"}}
        assert t.transform_response(err, model="gpt-4") == err

    def test_openai_to_anthropic_error_passthrough(self):
        """Anthropic error shapes are forwarded unchanged instead of being translated."""
        t = TranslationTransform("openai_to_anthropic")
        err1 = {"type": "error", "error": {"type": "authentication_error", "message": "bad"}}
        assert t.transform_response(err1) == err1
        err2 = {"error": {"message": "rate limit"}}
        assert t.transform_response(err2) == err2

    def test_stream_line_anthropic_to_openai(self):
        t = TranslationTransform("anthropic_to_openai")
        state = {}
        line = 'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}'
        result = t.transform_stream_line(line, state)
        assert isinstance(result, list)

    def test_stream_line_openai_to_anthropic(self):
        t = TranslationTransform("openai_to_anthropic")
        state = {}
        line = 'event: message_start'
        result = t.transform_stream_line(line, state)
        assert isinstance(result, list)


# ── Registry ─────────────────────────────────────────────────────────────


class TestRegistry:
    def test_known_transforms(self):
        assert "ghostkey" in REGISTRY
        assert "translation" in REGISTRY
        assert "llm_annotator" in REGISTRY

    def test_ghostkey_factory(self):
        t = REGISTRY["ghostkey"]({})
        assert t.name == "ghostkey"

    def test_translation_factory(self):
        t = REGISTRY["translation"]({"direction": "anthropic_to_openai"})
        assert t.name == "translation"

    def test_llm_annotator_factory(self):
        t = REGISTRY["llm_annotator"]({})
        assert t.name == "llm_annotator"
        assert isinstance(t, LLMAnnotatorTransform)
        assert hasattr(t, "annotate_request")
        assert hasattr(t, "annotate_response")

    def test_translation_factory_missing_direction(self):
        with pytest.raises(ValueError, match="direction"):
            REGISTRY["translation"]({})


# ── build_pipeline ──────────────────────────────────────────────────────


class TestBuildPipeline:
    def test_no_config_file(self, tmp_path):
        pipeline = build_pipeline(str(tmp_path / "nonexistent.yaml"))
        assert pipeline == []

    def test_empty_transforms(self, tmp_path):
        cfg = tmp_path / "malcolm.yaml"
        cfg.write_text(yaml.dump({"transforms": []}))
        assert build_pipeline(str(cfg)) == []

    def test_ghostkey_only(self, tmp_path):
        cfg = tmp_path / "malcolm.yaml"
        cfg.write_text(yaml.dump({"transforms": ["ghostkey"]}))
        pipeline = build_pipeline(str(cfg))
        assert len(pipeline) == 1
        assert pipeline[0].name == "ghostkey"

    def test_translation_with_config(self, tmp_path):
        cfg = tmp_path / "malcolm.yaml"
        cfg.write_text(yaml.dump({
            "transforms": [{"translation": {"direction": "anthropic_to_openai"}}],
        }))
        pipeline = build_pipeline(str(cfg))
        assert len(pipeline) == 1
        assert pipeline[0].name == "translation"

    def test_both_respects_order(self, tmp_path):
        cfg = tmp_path / "malcolm.yaml"
        cfg.write_text(yaml.dump({
            "transforms": [
                "ghostkey",
                {"translation": {"direction": "openai_to_anthropic"}},
            ],
        }))
        pipeline = build_pipeline(str(cfg))
        assert len(pipeline) == 2
        assert pipeline[0].name == "ghostkey"
        assert pipeline[1].name == "translation"

    def test_reverse_order(self, tmp_path):
        cfg = tmp_path / "malcolm.yaml"
        cfg.write_text(yaml.dump({
            "transforms": [
                {"translation": {"direction": "openai_to_anthropic"}},
                "ghostkey",
            ],
        }))
        pipeline = build_pipeline(str(cfg))
        assert pipeline[0].name == "translation"
        assert pipeline[1].name == "ghostkey"

    def test_unknown_transform_raises(self, tmp_path):
        cfg = tmp_path / "malcolm.yaml"
        cfg.write_text(yaml.dump({"transforms": ["nonexistent"]}))
        with pytest.raises(ValueError, match="Unknown transform.*nonexistent"):
            build_pipeline(str(cfg))


# ── Entry point discovery ───────────────────────────────────────────────


class _FakeTransform:
    def __init__(self, name):
        self.name = name

    def transform_request(self, body):
        return body

    def transform_response(self, body, model=""):
        return body

    def transform_stream_line(self, line, state):
        return [line]

    def rewrite_path(self, path):
        return path


def _make_factory(name):
    def factory(config):
        return _FakeTransform(name)
    return factory


class _FakeEntryPoint:
    def __init__(self, name, factory=None, load_error=None):
        self.name = name
        self._factory = factory
        self._load_error = load_error

    def load(self):
        if self._load_error is not None:
            raise self._load_error
        return self._factory


@pytest.fixture
def registry_snapshot():
    original = dict(transforms_module.REGISTRY)
    yield transforms_module.REGISTRY
    transforms_module.REGISTRY.clear()
    transforms_module.REGISTRY.update(original)


class TestEntryPointDiscovery:
    def test_external_plugin_registered(self, registry_snapshot, tmp_path):
        ep = _FakeEntryPoint("myplugin", factory=_make_factory("myplugin"))
        _discover_entry_points([ep])

        assert "myplugin" in registry_snapshot

        cfg = tmp_path / "malcolm.yaml"
        cfg.write_text(yaml.dump({"transforms": ["myplugin"]}))
        pipeline = build_pipeline(str(cfg))
        assert len(pipeline) == 1
        assert pipeline[0].name == "myplugin"

    def test_builtin_precedence(self, registry_snapshot, caplog):
        original_ghostkey = registry_snapshot["ghostkey"]
        hostile = _FakeEntryPoint("ghostkey", factory=_make_factory("hostile"))

        with caplog.at_level("WARNING", logger="malcolm.transforms"):
            _discover_entry_points([hostile])

        assert registry_snapshot["ghostkey"] is original_ghostkey
        assert any("ghostkey" in r.message for r in caplog.records)

    def test_external_collision_first_wins(self, registry_snapshot):
        first = _FakeEntryPoint("dup", factory=_make_factory("first"))
        second = _FakeEntryPoint("dup", factory=_make_factory("second"))

        _discover_entry_points([first, second])

        instance = registry_snapshot["dup"]({})
        assert instance.name == "first"

    def test_broken_plugin_skipped(self, registry_snapshot, caplog):
        broken = _FakeEntryPoint(
            "broken", load_error=RuntimeError("boom")
        )
        good = _FakeEntryPoint("good", factory=_make_factory("good"))

        with caplog.at_level("WARNING", logger="malcolm.transforms"):
            _discover_entry_points([broken, good])

        assert "broken" not in registry_snapshot
        assert "good" in registry_snapshot
        assert any(
            "broken" in r.message and "boom" in r.message
            for r in caplog.records
        )
