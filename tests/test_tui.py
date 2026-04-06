import sys

import pytest
from rich.text import Text

from malcolm.storage import RequestRecord, Storage
from malcolm.tui import (
    DetailScreen,
    GroupsScreen,
    MalcolmTUI,
    MessagesScreen,
    RequestsScreen,
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
    # Two requests in the same session (same model, close timestamps)
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

    # Same model, 10 min later — same session
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
async def anthropic_storage(storage):
    """Storage with Anthropic-format records."""
    record = RequestRecord(
        id="req-anth-001",
        timestamp="2026-01-01T00:00:00",
        model="claude-opus-4-6",
        stream=False,
        request_body={
            "model": "claude-opus-4-6",
            "system": "You are helpful",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            ],
        },
        response_body={
            "model": "claude-opus-4-6",
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hi from Claude!"}],
        },
        status_code=200,
        duration_ms=800.0,
    )
    await storage.save(record)

    # Streaming Anthropic response — 5 min later, same session
    stream_record = RequestRecord(
        id="req-anth-002",
        timestamp="2026-01-01T00:05:00",
        model="claude-opus-4-6",
        stream=True,
        request_body={
            "model": "claude-opus-4-6",
            "system": "Be concise",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "stream me"}]}],
        },
        response_chunks=[
            {"type": "message_start", "message": {"role": "assistant"}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello "}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "world!"}},
            {"type": "content_block_stop", "index": 0},
        ],
        status_code=200,
        duration_ms=600.0,
    )
    await storage.save(stream_record)

    # Tool call — different model, separate session
    tool_record = RequestRecord(
        id="req-anth-003",
        timestamp="2026-01-03T00:00:00",
        model="claude-opus-4-6",
        stream=False,
        request_body={
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "read a file"}],
        },
        response_body={
            "id": "resp-tool",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "claude-opus-4-6",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"function": {"name": "read_file"}, "type": "function", "id": "tc_1"}],
                },
                "finish_reason": "tool_calls",
            }],
        },
        status_code=200,
        duration_ms=300.0,
    )
    await storage.save(tool_record)
    return storage


# ---------------------------------------------------------------------------
# TUI app tests
# ---------------------------------------------------------------------------

async def test_tui_app_creates_with_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    app = MalcolmTUI(db_path=db_path)
    assert app._db_path == db_path


async def test_groups_screen_shows_groups(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        screen = app.screen
        assert isinstance(screen, GroupsScreen)
        table = screen.query_one("DataTable")
        # No session hints → each record is its own group
        assert table.row_count == 2


async def test_groups_screen_empty_db(storage):
    app = MalcolmTUI(db_path=storage._db_path)
    async with app.run_test() as pilot:
        table = app.screen.query_one("DataTable")
        assert table.row_count == 1  # placeholder row


async def test_drill_into_requests(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, RequestsScreen)
        table = screen.query_one("DataTable")
        # No session hints → each group has 1 request
        assert table.row_count == 1


async def test_drill_into_messages(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        await pilot.press("enter")  # → requests
        await pilot.pause()
        await pilot.press("enter")  # → messages of first request
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, MessagesScreen)


async def test_drill_into_detail(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        await pilot.press("enter")  # → requests
        await pilot.pause()
        await pilot.press("enter")  # → messages
        await pilot.pause()
        await pilot.press("enter")  # → detail
        await pilot.pause()
        assert isinstance(app.screen, DetailScreen)


async def test_back_navigation_escape(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, RequestsScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, GroupsScreen)


async def test_back_navigation_h(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, RequestsScreen)
        await pilot.press("h")
        await pilot.pause()
        assert isinstance(app.screen, GroupsScreen)


async def test_vim_navigation_l_selects(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        await pilot.press("l")
        await pilot.pause()
        assert isinstance(app.screen, RequestsScreen)


async def test_streaming_record_assembles_response(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        await pilot.press("enter")  # → requests
        await pilot.pause()
        # First row is req-002 (newest, streaming)
        await pilot.press("enter")  # → messages
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, MessagesScreen)
        assert table_has_text(screen, "chunk")


# ---------------------------------------------------------------------------
# Anthropic format tests
# ---------------------------------------------------------------------------

async def test_anthropic_response_extracted(anthropic_storage):
    app = MalcolmTUI(db_path=anthropic_storage._db_path)
    async with app.run_test() as pilot:
        # 3 groups (no session hints), newest first: req-anth-003, 002, 001
        # Navigate to req-anth-001 (third group)
        await pilot.press("j")
        await pilot.press("j")
        await pilot.press("enter")  # → requests (1 request)
        await pilot.pause()
        await pilot.press("enter")  # → messages
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, MessagesScreen)
        assert table_has_text(screen, "Hi from Claude!")


async def test_anthropic_streaming_assembled(anthropic_storage):
    app = MalcolmTUI(db_path=anthropic_storage._db_path)
    async with app.run_test() as pilot:
        # req-anth-002 is second group (streaming)
        await pilot.press("j")
        await pilot.press("enter")  # → requests (1 request)
        await pilot.pause()
        await pilot.press("enter")  # → messages
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, MessagesScreen)
        assert table_has_text(screen, "Hello world!")


async def test_tool_call_shows_tool_name(anthropic_storage):
    app = MalcolmTUI(db_path=anthropic_storage._db_path)
    async with app.run_test() as pilot:
        # First group is req-anth-003 (tool call, newest)
        await pilot.press("enter")  # → requests
        await pilot.pause()
        await pilot.press("enter")  # → messages
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, MessagesScreen)
        assert table_has_text(screen, "tool: read_file")


# ---------------------------------------------------------------------------
# Role styles
# ---------------------------------------------------------------------------

async def test_messages_have_role_styles(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        # Second group is req-001 (non-streaming, has system+user+assistant)
        await pilot.press("j")
        await pilot.press("enter")  # → requests (1 request)
        await pilot.pause()
        await pilot.press("enter")  # → messages
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, MessagesScreen)
        table = screen.query_one("DataTable")
        col_keys = list(table.columns.keys())
        role_col = col_keys[1]
        role_texts = []
        for row_key in table.rows:
            role_cell = table.get_cell(row_key, role_col)
            assert isinstance(role_cell, Text), f"Expected Text, got {type(role_cell)}"
            role_texts.append(str(role_cell))
        assert "user" in role_texts
        assert "assistant" in role_texts


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
            request_body={
                "model": "gpt-4o",
                "messages": [],
            },
            status_code=200,
            duration_ms=100.0,
        )
        await storage.save(record)

        await pilot.press("r")
        await pilot.pause()
        table = app.screen.query_one("DataTable")
        assert table.row_count == 1  # 1 group with 1 request
        assert table_has_text(app.screen, "gpt-4o")


# ---------------------------------------------------------------------------
# Subtitle updates
# ---------------------------------------------------------------------------

async def test_subtitle_updates_on_back(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        assert "Sessions" in app.sub_title
        await pilot.press("enter")
        await pilot.pause()
        assert "gpt-4o" in app.sub_title
        await pilot.press("escape")
        await pilot.pause()
        assert "Sessions" in app.sub_title


# ---------------------------------------------------------------------------
# Detail screen
# ---------------------------------------------------------------------------

async def test_detail_shows_json(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        await pilot.press("enter")  # → requests
        await pilot.pause()
        await pilot.press("enter")  # → messages
        await pilot.pause()
        await pilot.press("enter")  # → detail
        await pilot.pause()
        assert isinstance(app.screen, DetailScreen)


async def test_detail_back_to_messages(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, DetailScreen)
        await pilot.press("h")
        await pilot.pause()
        assert isinstance(app.screen, MessagesScreen)


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
