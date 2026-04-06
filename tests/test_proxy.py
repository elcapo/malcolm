import json

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from malcolm.config import Settings
from malcolm.formats import assemble_openai_chunks
from malcolm.proxy import _build_target_url, forward_request, forward_request_stream
from malcolm.storage import NullStorage, RequestRecord


# --- Unit tests for helpers ---


@pytest.mark.parametrize(
    "base_url, path, expected",
    [
        # No base path — just append
        ("https://api.example.com", "/v1/chat/completions", "https://api.example.com/v1/chat/completions"),
        ("https://api.example.com/", "/v1/chat/completions", "https://api.example.com/v1/chat/completions"),
        ("https://api.anthropic.com", "/v1/messages", "https://api.anthropic.com/v1/messages"),
        ("http://localhost:11434", "/api/chat", "http://localhost:11434/api/chat"),
        # Base path /v1, client includes /v1 — no duplication
        ("http://localhost:11434/v1", "/v1/chat/completions", "http://localhost:11434/v1/chat/completions"),
        ("http://localhost:11434/v1", "/v1/messages", "http://localhost:11434/v1/messages"),
        ("http://localhost:11434/v1/", "/v1/chat/completions", "http://localhost:11434/v1/chat/completions"),
        # Base path /v1, client omits /v1 — prepend it
        ("http://localhost:11434/v1", "/chat/completions", "http://localhost:11434/v1/chat/completions"),
        ("http://localhost:11434/v1", "/messages", "http://localhost:11434/v1/messages"),
        # Deeper base path
        ("https://gateway.example.com/api/v1", "/v1/chat/completions", "https://gateway.example.com/api/v1/v1/chat/completions"),
        ("https://gateway.example.com/api/v1", "/chat/completions", "https://gateway.example.com/api/v1/chat/completions"),
    ],
)
def test_build_target_url(base_url, path, expected):
    assert _build_target_url(base_url, path) == expected


def testassemble_openai_chunks_empty():
    assert assemble_openai_chunks([]) == {}


def testassemble_openai_chunks_text():
    chunks = [
        {"id": "c1", "created": 100, "model": "gpt-4", "choices": [{"delta": {"role": "assistant", "content": ""}, "index": 0}]},
        {"id": "c1", "created": 100, "model": "gpt-4", "choices": [{"delta": {"content": "Hello"}, "index": 0}]},
        {"id": "c1", "created": 100, "model": "gpt-4", "choices": [{"delta": {"content": " world"}, "index": 0}]},
        {"id": "c1", "created": 100, "model": "gpt-4", "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
    ]

    result = assemble_openai_chunks(chunks)

    assert result["id"] == "c1"
    assert result["model"] == "gpt-4"
    assert result["choices"][0]["message"]["content"] == "Hello world"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["total_tokens"] == 7


def testassemble_openai_chunks_tool_calls():
    chunks = [
        {"id": "c1", "created": 100, "model": "gpt-4", "choices": [{"delta": {"role": "assistant", "tool_calls": [{"index": 0, "id": "tc1", "type": "function", "function": {"name": "get_weather", "arguments": ""}}]}, "index": 0}]},
        {"id": "c1", "created": 100, "model": "gpt-4", "choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]}, "index": 0}]},
        {"id": "c1", "created": 100, "model": "gpt-4", "choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"NYC"}'}}]}, "index": 0}]},
        {"id": "c1", "created": 100, "model": "gpt-4", "choices": [{"delta": {}, "index": 0, "finish_reason": "tool_calls"}]},
    ]

    result = assemble_openai_chunks(chunks)

    tc = result["choices"][0]["message"]["tool_calls"][0]
    assert tc["id"] == "tc1"
    assert tc["function"]["name"] == "get_weather"
    assert tc["function"]["arguments"] == '{"city":"NYC"}'
    assert result["choices"][0]["finish_reason"] == "tool_calls"


# --- Integration tests with a fake backend ---


def _make_fake_backend(response_body: dict | None = None, stream_chunks: list[str] | None = None):
    """Create a fake backend FastAPI app."""
    backend = FastAPI()

    @backend.post("/v1/chat/completions")
    async def chat(request: Request):
        if stream_chunks is not None:
            from fastapi.responses import StreamingResponse

            async def generate():
                for chunk in stream_chunks:
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")
        return response_body or {"id": "test", "choices": []}

    return backend


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("MALCOLM_TARGET_URL", "http://testserver")
    return Settings()


@pytest.fixture
def null_storage():
    return NullStorage()


def test_forward_non_streaming(settings, null_storage):
    response_body = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }
    backend = _make_fake_backend(response_body=response_body)

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def proxy(request: Request):
        body = await request.json()
        transport = httpx.ASGITransport(app=backend)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await forward_request(body, request, client, settings, null_storage)

    test_client = TestClient(app)
    resp = test_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "chatcmpl-test"
    assert data["choices"][0]["message"]["content"] == "Hi!"


def test_forward_streaming(settings, null_storage):
    chunks = [
        json.dumps({"id": "c1", "created": 100, "model": "gpt-4", "choices": [{"delta": {"role": "assistant", "content": ""}, "index": 0}]}),
        json.dumps({"id": "c1", "created": 100, "model": "gpt-4", "choices": [{"delta": {"content": "Hello"}, "index": 0}]}),
        json.dumps({"id": "c1", "created": 100, "model": "gpt-4", "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]}),
    ]
    backend = _make_fake_backend(stream_chunks=chunks)
    transport = httpx.ASGITransport(app=backend)
    # Client must outlive the streaming response, so create it in lifespan
    shared_client = httpx.AsyncClient(transport=transport, base_url="http://testserver")

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def proxy(request: Request):
        body = await request.json()
        return await forward_request_stream(body, request, shared_client, settings, null_storage)

    test_client = TestClient(app)
    resp = test_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}], "stream": True},
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"

    lines = [l for l in resp.text.strip().split("\n") if l.startswith("data: ")]
    assert len(lines) >= 3  # 3 chunks + [DONE]


def test_forward_with_anthropic_to_openai_translation(null_storage, monkeypatch):
    """Send Anthropic-format request, verify OpenAI response translated back."""
    monkeypatch.setenv("MALCOLM_TARGET_URL", "http://testserver")
    monkeypatch.setenv("MALCOLM_TRANSLATE", "anthropic_to_openai")
    translate_settings = Settings()

    openai_response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4.1",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }
    backend = _make_fake_backend(response_body=openai_response)

    app = FastAPI()

    @app.post("/v1/messages")
    async def proxy(request: Request):
        body = await request.json()
        transport = httpx.ASGITransport(app=backend)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await forward_request(body, request, client, translate_settings, null_storage)

    test_client = TestClient(app)
    # Send Anthropic-format request
    resp = test_client.post(
        "/v1/messages",
        json={
            "model": "gpt-4.1",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    # Should be Anthropic-format response
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["content"][0]["type"] == "text"
    assert data["content"][0]["text"] == "Hi!"
    assert data["stop_reason"] == "end_turn"


def test_forward_with_openai_to_anthropic_translation(null_storage, monkeypatch):
    """Send OpenAI-format request, verify Anthropic response translated back."""
    monkeypatch.setenv("MALCOLM_TARGET_URL", "http://testserver")
    monkeypatch.setenv("MALCOLM_TRANSLATE", "openai_to_anthropic")
    translate_settings = Settings()

    # Fake Anthropic backend
    anthropic_backend = FastAPI()

    @anthropic_backend.post("/v1/messages")
    async def messages(request: Request):
        return {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def proxy(request: Request):
        body = await request.json()
        transport = httpx.ASGITransport(app=anthropic_backend)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await forward_request(body, request, client, translate_settings, null_storage)

    test_client = TestClient(app)
    resp = test_client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-sonnet-4-20250514",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    # Should be OpenAI-format response
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "Hello!"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"]["total_tokens"] == 7


def test_forward_backend_error(settings, null_storage):
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def proxy(request: Request):
        body = await request.json()
        # Point to a non-existent server
        async with httpx.AsyncClient() as client:
            settings.target_url = "http://localhost:1"  # should fail to connect
            return await forward_request(body, request, client, settings, null_storage)

    test_client = TestClient(app)
    resp = test_client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert resp.status_code == 502
    assert "proxy_error" in resp.json()["error"]["type"]
