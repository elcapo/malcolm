import uuid

import pytest

from malcolm.storage import NullStorage, RequestRecord, Storage


@pytest.fixture
async def storage(tmp_path):
    db_path = str(tmp_path / "test.db")
    s = Storage(db_path)
    await s.init()
    yield s
    await s.close()


def _make_record(**kwargs) -> RequestRecord:
    defaults = {
        "id": str(uuid.uuid4()),
        "model": "gpt-4",
        "stream": False,
        "request_body": {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
        },
    }
    defaults.update(kwargs)
    return RequestRecord(**defaults)


async def test_save_and_get(storage):
    record = _make_record()
    await storage.save(record)

    result = await storage.get(record.id)
    assert result is not None
    assert result["id"] == record.id
    assert result["model"] == "gpt-4"
    assert result["request_body"]["messages"][0]["content"] == "hello"


async def test_save_with_response(storage):
    record = _make_record(
        response_body={"id": "chatcmpl-1", "choices": [{"message": {"content": "hi"}}]},
        status_code=200,
        duration_ms=150.5,
    )
    await storage.save(record)

    result = await storage.get(record.id)
    assert result["status_code"] == 200
    assert result["duration_ms"] == 150.5
    assert result["response_body"]["choices"][0]["message"]["content"] == "hi"


async def test_save_streaming_chunks(storage):
    chunks = [
        {"choices": [{"delta": {"content": "hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
    ]
    record = _make_record(stream=True, response_chunks=chunks)
    await storage.save(record)

    result = await storage.get(record.id)
    assert result["stream"] == 1  # SQLite stores as integer
    assert len(result["response_chunks"]) == 2


async def test_list_recent(storage):
    for i in range(5):
        await storage.save(_make_record(model=f"model-{i}"))

    results = await storage.list_recent(limit=3)
    assert len(results) == 3


async def test_list_recent_empty(storage):
    results = await storage.list_recent()
    assert results == []


async def test_get_nonexistent(storage):
    result = await storage.get("nonexistent-id")
    assert result is None


async def test_delete(storage):
    record = _make_record()
    await storage.save(record)

    deleted = await storage.delete(record.id)
    assert deleted is True

    result = await storage.get(record.id)
    assert result is None


async def test_delete_nonexistent(storage):
    deleted = await storage.delete("nonexistent-id")
    assert deleted is False


async def test_save_with_error(storage):
    record = _make_record(error="Connection refused", status_code=None)
    await storage.save(record)

    result = await storage.get(record.id)
    assert result["error"] == "Connection refused"


# NullStorage tests


async def test_null_storage_save():
    ns = NullStorage()
    await ns.init()
    await ns.save(_make_record())  # should not raise
    await ns.close()


async def test_null_storage_list_recent():
    ns = NullStorage()
    result = await ns.list_recent()
    assert result == []


async def test_null_storage_get():
    ns = NullStorage()
    result = await ns.get("any-id")
    assert result is None


async def test_null_storage_delete():
    ns = NullStorage()
    result = await ns.delete("any-id")
    assert result is False
