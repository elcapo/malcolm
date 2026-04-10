import uuid

import pytest

from malcolm.storage import NullStorage, RequestRecord, Storage, TransformRecord


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


# list_page_full tests


async def test_list_page_full_basic(storage):
    for i in range(5):
        await storage.save(_make_record(
            timestamp=f"2026-01-01T00:{i:02d}:00",
            model=f"model-{i}",
        ))

    results = await storage.list_page_full(page_size=3)
    assert len(results) == 3
    # Should be newest first
    assert results[0]["model"] == "model-4"
    assert results[1]["model"] == "model-3"
    assert results[2]["model"] == "model-2"
    # Should include request_body as dict
    assert isinstance(results[0]["request_body"], dict)


async def test_list_page_full_cursor(storage):
    for i in range(5):
        await storage.save(_make_record(
            timestamp=f"2026-01-01T00:{i:02d}:00",
            model=f"model-{i}",
        ))

    # First page
    page1 = await storage.list_page_full(page_size=2)
    assert len(page1) == 2
    assert page1[0]["model"] == "model-4"

    # Second page using cursor
    cursor = page1[-1]["timestamp"]
    page2 = await storage.list_page_full(page_size=2, before=cursor)
    assert len(page2) == 2
    assert page2[0]["model"] == "model-2"

    # Third page
    cursor2 = page2[-1]["timestamp"]
    page3 = await storage.list_page_full(page_size=2, before=cursor2)
    assert len(page3) == 1
    assert page3[0]["model"] == "model-0"


async def test_list_page_full_empty(storage):
    results = await storage.list_page_full()
    assert results == []


async def test_null_storage_list_page_full():
    ns = NullStorage()
    result = await ns.list_page_full()
    assert result == []


# TransformRecord tests


async def test_save_and_get_transform(storage):
    record = _make_record()
    await storage.save(record)

    transform = TransformRecord(
        request_id=record.id,
        transform_type="ghostkey",
        request_body={"model": "gpt-4", "messages": [{"role": "user", "content": "obfuscated"}]},
        response_body={"choices": [{"message": {"content": "restored"}}]},
    )
    await storage.save_transform(transform)

    transforms = await storage.get_transforms(record.id)
    assert len(transforms) == 1
    assert transforms[0]["transform_type"] == "ghostkey"
    assert transforms[0]["request_body"]["messages"][0]["content"] == "obfuscated"
    assert transforms[0]["response_body"]["choices"][0]["message"]["content"] == "restored"


async def test_save_multiple_transforms(storage):
    record = _make_record()
    await storage.save(record)

    await storage.save_transform(TransformRecord(
        request_id=record.id,
        transform_type="ghostkey",
        request_body={"obfuscated": True},
    ))
    await storage.save_transform(TransformRecord(
        request_id=record.id,
        transform_type="translation",
        request_body={"translated": True},
    ))

    transforms = await storage.get_transforms(record.id)
    assert len(transforms) == 2
    types = {t["transform_type"] for t in transforms}
    assert types == {"ghostkey", "translation"}


async def test_get_includes_transforms(storage):
    record = _make_record()
    await storage.save(record)
    await storage.save_transform(TransformRecord(
        request_id=record.id,
        transform_type="ghostkey",
        request_body={"obfuscated": True},
    ))

    result = await storage.get(record.id)
    assert "transforms" in result
    assert len(result["transforms"]) == 1
    assert result["transforms"][0]["transform_type"] == "ghostkey"


async def test_get_no_transforms(storage):
    record = _make_record()
    await storage.save(record)

    result = await storage.get(record.id)
    assert result["transforms"] == []


async def test_transform_cascade_delete(storage):
    record = _make_record()
    await storage.save(record)
    await storage.save_transform(TransformRecord(
        request_id=record.id,
        transform_type="ghostkey",
        request_body={"obfuscated": True},
    ))

    await storage.delete(record.id)
    transforms = await storage.get_transforms(record.id)
    assert transforms == []


async def test_transform_upsert(storage):
    record = _make_record()
    await storage.save(record)

    await storage.save_transform(TransformRecord(
        request_id=record.id,
        transform_type="ghostkey",
        request_body={"version": 1},
    ))
    await storage.save_transform(TransformRecord(
        request_id=record.id,
        transform_type="ghostkey",
        request_body={"version": 2},
    ))

    transforms = await storage.get_transforms(record.id)
    assert len(transforms) == 1
    assert transforms[0]["request_body"]["version"] == 2


async def test_transform_with_chunks(storage):
    record = _make_record(stream=True)
    await storage.save(record)

    chunks = [{"choices": [{"delta": {"content": "hi"}}]}]
    await storage.save_transform(TransformRecord(
        request_id=record.id,
        transform_type="translation",
        response_chunks=chunks,
    ))

    transforms = await storage.get_transforms(record.id)
    assert len(transforms[0]["response_chunks"]) == 1


async def test_null_storage_save_transform():
    ns = NullStorage()
    await ns.save_transform(TransformRecord(
        request_id="any", transform_type="ghostkey",
    ))


async def test_null_storage_get_transforms():
    ns = NullStorage()
    result = await ns.get_transforms("any-id")
    assert result == []


# Annotation tests


async def test_save_and_get_annotations(storage):
    from malcolm.transforms._base import Annotation

    record = _make_record()
    await storage.save(record)

    annotations = [
        Annotation(key="model", value="gpt-4", category="metadata", display="badge", source="request"),
        Annotation(key="session_id", value="abc123", category="metadata", display="kv", source="request"),
    ]
    await storage.save_annotations(record.id, "llm_annotator", annotations)

    result = await storage.get_annotations(record.id)
    assert len(result) == 2
    keys = {a["key"] for a in result}
    assert keys == {"model", "session_id"}
    for a in result:
        assert a["transform_name"] == "llm_annotator"
        assert a["source"] == "request"


async def test_get_includes_annotations(storage):
    from malcolm.transforms._base import Annotation

    record = _make_record()
    await storage.save(record)
    await storage.save_annotations(record.id, "llm_annotator", [
        Annotation(key="model", value="gpt-4", category="metadata", display="badge"),
    ])

    result = await storage.get(record.id)
    assert "annotations" in result
    assert len(result["annotations"]) == 1
    assert result["annotations"][0]["key"] == "model"


async def test_annotations_cascade_delete(storage):
    from malcolm.transforms._base import Annotation

    record = _make_record()
    await storage.save(record)
    await storage.save_annotations(record.id, "llm_annotator", [
        Annotation(key="model", value="gpt-4", category="metadata", display="badge"),
    ])

    await storage.delete(record.id)
    result = await storage.get_annotations(record.id)
    assert result == []


async def test_list_page_with_badges(storage):
    from malcolm.transforms._base import Annotation

    for i in range(3):
        record = _make_record(
            timestamp=f"2026-01-01T00:{i:02d}:00",
            model=f"model-{i}",
        )
        await storage.save(record)
        if i < 2:  # Only annotate first 2 records
            await storage.save_annotations(record.id, "llm_annotator", [
                Annotation(key="model", value=f"model-{i}", category="metadata", display="badge"),
            ])

    results = await storage.list_page_with_badges(page_size=10)
    assert len(results) == 3
    # Most recent first (model-2) has no badges
    assert results[0]["badges"] == {}
    # Second and third have badges
    assert results[1]["badges"]["model"] == "model-1"
    assert results[2]["badges"]["model"] == "model-0"


async def test_list_page_with_badges_empty(storage):
    results = await storage.list_page_with_badges()
    assert results == []


async def test_null_storage_annotations():
    from malcolm.transforms._base import Annotation

    ns = NullStorage()
    await ns.save_annotations("any", "t", [Annotation("k", "v")])
    assert await ns.get_annotations("any") == []
    assert await ns.list_page_with_badges() == []
