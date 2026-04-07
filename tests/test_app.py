import pytest
from fastapi.testclient import TestClient

from malcolm.app import create_app
from malcolm.config import Settings


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("MALCOLM_TARGET_URL", "http://testserver")
    monkeypatch.setenv("MALCOLM_DB_PATH", str(tmp_path / "test.db"))
    return Settings()


@pytest.fixture
def client(settings):
    app = create_app(settings)
    with TestClient(app) as c:
        yield c



def test_head_health_check(client):
    resp = client.head("/")
    assert resp.status_code == 200



def test_chat_completions_endpoint_exists(client):
    # Should return 502 (backend unreachable), not 404
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 502


def test_chat_completions_without_v1_prefix(client):
    resp = client.post(
        "/chat/completions",
        json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 502


def test_anthropic_messages_endpoint(client):
    """Catch-all should forward Anthropic API requests too."""
    resp = client.post(
        "/v1/messages",
        json={"model": "claude-sonnet-4-20250514", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10},
    )
    assert resp.status_code == 502  # backend unreachable, but not 404


def test_catch_all_get(client):
    """GET requests to unknown paths should be forwarded, not 404."""
    resp = client.get("/v1/models")
    assert resp.status_code == 502  # backend unreachable, but not 404
