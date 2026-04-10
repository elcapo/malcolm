import sys

import pytest
from rich.text import Text

from malcolm.storage import RequestRecord, Storage
from malcolm.transforms._base import Annotation
from malcolm.tui import (
    AnnotationsScreen,
    ContentScreen,
    MalcolmTUI,
    RequestListScreen,
)


@pytest.fixture
async def storage(tmp_path):
    db_path = str(tmp_path / "test.db")
    s = Storage(db_path)
    await s.init()
    yield s
    await s.close()


@pytest.fixture
async def populated_storage(storage):
    record = RequestRecord(
        id="req-001",
        timestamp="2026-01-01T00:00:00",
        model="gpt-4o",
        stream=False,
        request_body={
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
            ],
        },
        response_body={
            "id": "resp-001",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi there!"},
                    "finish_reason": "stop",
                }
            ],
        },
        status_code=200,
        duration_ms=1234.5,
    )
    await storage.save(record)

    stream_record = RequestRecord(
        id="req-002",
        timestamp="2026-01-01T00:10:00",
        model="gpt-4o",
        stream=True,
        request_body={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Stream test"}],
        },
        response_chunks=[{"choices": [{"delta": {"content": "chunk"}}]}],
        status_code=200,
        duration_ms=500.0,
    )
    await storage.save(stream_record)
    return storage


@pytest.fixture
async def annotated_storage(populated_storage):
    """Storage with request and response annotations on first record."""
    await populated_storage.save_annotations("req-001", "llm_annotator", [
        Annotation(key="model", value="gpt-4o", category="metadata", display="badge", source="request"),
        Annotation(key="session_id", value="sess-abc", category="metadata", display="kv", source="request"),
        Annotation(key="system_prompt", value="You are helpful.", category="content", display="text", source="request"),
        Annotation(key="assistant_message.0", value="Hi there!", category="content", display="text", source="response"),
        Annotation(key="input_tokens", value="10", category="usage", display="kv", source="response"),
    ])
    return populated_storage


# ---------------------------------------------------------------------------
# TUI app tests
# ---------------------------------------------------------------------------

async def test_tui_app_creates_with_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    app = MalcolmTUI(db_path=db_path)
    assert app._db_path == db_path


async def test_request_list_shows_requests(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        screen = app.screen
        assert isinstance(screen, RequestListScreen)
        table = screen.query_one("DataTable")
        assert table.row_count == 2


async def test_request_list_empty_db(storage):
    app = MalcolmTUI(db_path=storage._db_path)
    async with app.run_test() as pilot:
        table = app.screen.query_one("DataTable")
        assert table.row_count == 1  # placeholder row


async def test_request_list_shows_uuid(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        assert table_has_text(app.screen, "req-002")


async def test_request_list_shows_status(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        assert table_has_text(app.screen, "200")


# ---------------------------------------------------------------------------
# Badge annotations in list
# ---------------------------------------------------------------------------

async def test_badge_annotations_create_columns(annotated_storage):
    app = MalcolmTUI(db_path=annotated_storage._db_path)
    async with app.run_test() as pilot:
        table = app.screen.query_one("DataTable")
        col_labels = [str(table.columns[k].label) for k in table.columns]
        assert "Model" in col_labels
        assert table_has_text(app.screen, "gpt-4o")


async def test_mixed_annotations_no_crash(annotated_storage):
    """Records without badges still render fine alongside annotated ones."""
    app = MalcolmTUI(db_path=annotated_storage._db_path)
    async with app.run_test() as pilot:
        table = app.screen.query_one("DataTable")
        assert table.row_count == 2


# ---------------------------------------------------------------------------
# Navigation: list -> annotations -> content
# ---------------------------------------------------------------------------

async def test_drill_into_annotations(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, AnnotationsScreen)


async def test_annotations_shows_raw_json_entries(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        # Navigate to req-001 (second row) which has both request and response body
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, AnnotationsScreen)
        # Should have at least raw request and raw response entries
        assert table_has_text(screen, "raw request")
        assert table_has_text(screen, "raw response")


async def test_annotations_shows_annotation_rows(annotated_storage):
    app = MalcolmTUI(db_path=annotated_storage._db_path)
    async with app.run_test() as pilot:
        # Navigate to req-001 (second row)
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, AnnotationsScreen)
        assert table_has_text(screen, "model")
        assert table_has_text(screen, "system_prompt")


async def test_annotations_grouped_by_source(annotated_storage):
    app = MalcolmTUI(db_path=annotated_storage._db_path)
    async with app.run_test() as pilot:
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, AnnotationsScreen)
        assert table_has_text(screen, "request")
        assert table_has_text(screen, "response")


async def test_drill_into_content(annotated_storage):
    app = MalcolmTUI(db_path=annotated_storage._db_path)
    async with app.run_test() as pilot:
        await pilot.press("down")
        await pilot.press("enter")  # -> annotations
        await pilot.pause()
        await pilot.press("enter")  # -> content of first annotation
        await pilot.pause()
        assert isinstance(app.screen, ContentScreen)


async def test_drill_into_raw_json(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        await pilot.press("down")      # navigate to req-001
        await pilot.press("enter")  # -> annotations
        await pilot.pause()
        await pilot.press("enter")  # -> first item (raw request)
        await pilot.pause()
        assert isinstance(app.screen, ContentScreen)


# ---------------------------------------------------------------------------
# Back navigation
# ---------------------------------------------------------------------------

async def test_back_from_annotations(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, AnnotationsScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, RequestListScreen)


async def test_back_from_content(annotated_storage):
    app = MalcolmTUI(db_path=annotated_storage._db_path)
    async with app.run_test() as pilot:
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, ContentScreen)
        await pilot.press("left")
        await pilot.pause()
        assert isinstance(app.screen, AnnotationsScreen)


async def test_enter_selects(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, AnnotationsScreen)


# ---------------------------------------------------------------------------
# Reload
# ---------------------------------------------------------------------------

async def test_reload_picks_up_new_records(storage):
    app = MalcolmTUI(db_path=storage._db_path)
    async with app.run_test() as pilot:
        table = app.screen.query_one("DataTable")
        assert table.row_count == 1  # placeholder

        record = RequestRecord(
            id="req-new",
            model="gpt-4o",
            request_body={"model": "gpt-4o", "messages": []},
            status_code=200,
            duration_ms=100.0,
        )
        await storage.save(record)

        await pilot.press("r")
        await pilot.pause()
        table = app.screen.query_one("DataTable")
        assert table.row_count == 1


# ---------------------------------------------------------------------------
# Follow mode
# ---------------------------------------------------------------------------

async def test_follow_toggle(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        screen = app.screen
        assert isinstance(screen, RequestListScreen)
        assert not screen._following

        await pilot.press("f")
        assert screen._following
        assert "FOLLOW" in app.sub_title

        await pilot.press("f")
        assert not screen._following
        assert "FOLLOW" not in app.sub_title


# ---------------------------------------------------------------------------
# Subtitle updates
# ---------------------------------------------------------------------------

async def test_subtitle_updates_on_back(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        assert "Requests" in app.sub_title
        await pilot.press("enter")
        await pilot.pause()
        assert "Detail" in app.sub_title
        await pilot.press("escape")
        await pilot.pause()
        assert "Requests" in app.sub_title


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def test_cli_tui_dispatch(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["malcolm", "tui", "--db-path", "/tmp/test.db"])
    from malcolm.cli import _parse_tui_args

    assert _parse_tui_args() == "/tmp/test.db"


async def test_cli_tui_dispatch_no_args(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["malcolm", "tui"])
    from malcolm.cli import _parse_tui_args

    assert _parse_tui_args() is None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def table_has_text(screen, text: str) -> bool:
    """Check if any cell in the screen's DataTable contains the given text."""
    table = screen.query_one("DataTable")
    for row_key in table.rows:
        for col_key in table.columns:
            cell = table.get_cell(row_key, col_key)
            if text.lower() in str(cell).lower():
                return True
    return False
