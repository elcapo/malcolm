import uuid

import pytest
from fastapi.testclient import TestClient

from malcolm.app import create_app
from malcolm.config import Settings
from malcolm.storage import RequestRecord


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


async def _insert_record(storage, **kwargs):
    defaults = {
        "id": str(uuid.uuid4()),
        "model": "gpt-4",
        "stream": False,
        "request_body": {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]},
        "response_body": {"id": "r1", "choices": [{"message": {"content": "hi"}}]},
        "status_code": 200,
        "duration_ms": 100.0,
    }
    defaults.update(kwargs)
    record = RequestRecord(**defaults)
    await storage.save(record)
    return record


def test_logs_page_with_records(client):
    import asyncio
    storage = client.app.state.storage
    loop = asyncio.get_event_loop()
    record = loop.run_until_complete(_insert_record(storage))

    resp = client.get("/logs")
    assert resp.status_code == 200
    assert record.id[:8] in resp.text
    assert "gpt-4" in resp.text


def test_log_detail_page(client):
    import asyncio
    storage = client.app.state.storage
    loop = asyncio.get_event_loop()
    record = loop.run_until_complete(_insert_record(storage))

    resp = client.get(f"/logs/{record.id}")
    assert resp.status_code == 200
    assert record.id in resp.text
    assert "gpt-4" in resp.text
    assert "hello" in resp.text  # request body content


def test_api_logs_with_records(client):
    import asyncio
    storage = client.app.state.storage
    loop = asyncio.get_event_loop()
    loop.run_until_complete(_insert_record(storage, model="claude-3"))

    resp = client.get("/api/logs")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    assert data[0]["model"] == "claude-3"


def test_api_log_detail(client):
    import asyncio
    storage = client.app.state.storage
    loop = asyncio.get_event_loop()
    record = loop.run_until_complete(_insert_record(storage))

    resp = client.get(f"/api/logs/{record.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == record.id
    assert data["request_body"]["messages"][0]["content"] == "hello"
    assert data["response_body"]["choices"][0]["message"]["content"] == "hi"


def test_log_detail_with_error(client):
    import asyncio
    storage = client.app.state.storage
    loop = asyncio.get_event_loop()
    record = loop.run_until_complete(
        _insert_record(storage, error="Connection refused", status_code=502)
    )

    resp = client.get(f"/logs/{record.id}")
    assert resp.status_code == 200
    assert "Connection refused" in resp.text
