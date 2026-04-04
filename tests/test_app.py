import json

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


@pytest.fixture
def client_no_storage(monkeypatch, tmp_path):
    monkeypatch.setenv("MALCOLM_TARGET_URL", "http://testserver")
    monkeypatch.setenv("MALCOLM_STORAGE_ENABLED", "false")
    settings = Settings()
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


def test_root_redirects_to_logs(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == "/logs"


def test_logs_page_empty(client):
    resp = client.get("/logs")
    assert resp.status_code == 200
    assert "malcolm" in resp.text
    assert "No requests logged yet" in resp.text


def test_api_logs_empty(client):
    resp = client.get("/api/logs")
    assert resp.status_code == 200
    assert resp.json() == []


def test_api_log_detail_not_found(client):
    resp = client.get("/api/logs/nonexistent")
    assert resp.status_code == 404


def test_log_detail_not_found(client):
    resp = client.get("/logs/nonexistent")
    assert resp.status_code == 404


def test_storage_disabled(client_no_storage):
    resp = client_no_storage.get("/api/logs")
    assert resp.status_code == 200
    assert resp.json() == []


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
