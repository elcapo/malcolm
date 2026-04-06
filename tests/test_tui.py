import json
import sys

import pytest
from rich.text import Text

from malcolm.storage import RequestRecord, Storage
from malcolm.tui import (
    DetailScreen,
    MalcolmTUI,
    MessagesScreen,
    SessionsScreen,
    _assemble_anthropic_chunks,
    _extract_assistant_message,
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
    # Older record — appears second in DESC order
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
            "metadata": {"user_id": '{"session_id": "sess-001"}'},
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

    # Newer record — appears first in DESC order
    stream_record = RequestRecord(
        id="req-002",
        timestamp="2026-01-02T00:00:00",
        model="gpt-4o",
        stream=True,
        request_body={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Stream test"}],
            "metadata": {"user_id": '{"session_id": "sess-001"}'},
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
    # Non-streaming Anthropic response
    record = RequestRecord(
        id="req-anth-001",
        timestamp="2026-01-01T00:00:00",
        model="claude-opus-4-6",
        stream=False,
        request_body={
            "model": "claude-opus-4-6",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "<system-reminder>hook output</system-reminder>"},
                    {"type": "text", "text": "hello"},
                ]},
            ],
            "metadata": {"user_id": '{"session_id": "sess-anth-001"}'},
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

    # Streaming Anthropic response (SSE chunks)
    stream_record = RequestRecord(
        id="req-anth-002",
        timestamp="2026-01-02T00:00:00",
        model="claude-opus-4-6",
        stream=True,
        request_body={
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "stream me"}],
            "metadata": {"user_id": '{"session_id": "sess-anth-002"}'},
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

    # Tool call response (content: null)
    tool_record = RequestRecord(
        id="req-anth-003",
        timestamp="2026-01-03T00:00:00",
        model="claude-opus-4-6",
        stream=False,
        request_body={
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "read a file"}],
            "metadata": {"user_id": '{"session_id": "sess-anth-003"}'},
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
# Unit tests for helper functions
# ---------------------------------------------------------------------------

class TestExtractAssistantMessage:
    def test_openai_format(self):
        resp = {
            "choices": [{"message": {"role": "assistant", "content": "Hi"}}],
        }
        msg = _extract_assistant_message(resp)
        assert msg["content"] == "Hi"

    def test_openai_format_null_content_with_tool_calls(self):
        resp = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"function": {"name": "foo"}}],
                },
            }],
        }
        msg = _extract_assistant_message(resp)
        assert msg is not None
        assert msg["tool_calls"][0]["function"]["name"] == "foo"

    def test_openai_format_null_content_no_tools(self):
        resp = {
            "choices": [{"message": {"role": "assistant", "content": None}}],
        }
        msg = _extract_assistant_message(resp)
        assert msg is None

    def test_anthropic_format(self):
        resp = {
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
        }
        msg = _extract_assistant_message(resp)
        assert msg["role"] == "assistant"
        assert msg["content"][0]["text"] == "Hello!"

    def test_empty_dict(self):
        assert _extract_assistant_message({}) is None

    def test_not_a_dict(self):
        assert _extract_assistant_message(None) is None


class TestAssembleAnthropicChunks:
    def test_anthropic_text_deltas(self):
        chunks = [
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello "}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "world!"}},
        ]
        msg = _assemble_anthropic_chunks(chunks)
        assert msg == {"role": "assistant", "content": "Hello world!"}

    def test_openai_format_fallback(self):
        chunks = [
            {"choices": [{"delta": {"content": "Hi "}}]},
            {"choices": [{"delta": {"content": "there"}}]},
        ]
        msg = _assemble_anthropic_chunks(chunks)
        assert msg == {"role": "assistant", "content": "Hi there"}

    def test_empty_chunks(self):
        assert _assemble_anthropic_chunks([]) is None

    def test_no_text_content(self):
        chunks = [
            {"type": "message_start", "message": {"role": "assistant"}},
            {"type": "content_block_stop", "index": 0},
        ]
        assert _assemble_anthropic_chunks(chunks) is None


# ---------------------------------------------------------------------------
# TUI app tests
# ---------------------------------------------------------------------------

async def test_tui_app_creates_with_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    app = MalcolmTUI(db_path=db_path)
    assert app._db_path == db_path


async def test_sessions_screen_shows_sessions(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        screen = app.screen
        assert isinstance(screen, SessionsScreen)
        table = screen.query_one("DataTable")
        # Both records share sess-001, so 1 session row
        assert table.row_count == 1


async def test_requests_screen_empty_db(storage):
    app = MalcolmTUI(db_path=storage._db_path)
    async with app.run_test() as pilot:
        table = app.screen.query_one("DataTable")
        assert table.row_count == 1  # placeholder row


async def test_drill_into_messages(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        # One session row — enters the latest request (req-002, streaming)
        await pilot.press("enter")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, MessagesScreen)
        table = screen.query_one("DataTable")
        # req-002 has 1 request message + 1 assembled assistant response = 2
        assert table.row_count == 2


async def test_drill_into_detail(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, DetailScreen)


async def test_back_navigation_escape(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, MessagesScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, SessionsScreen)


async def test_back_navigation_h(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, MessagesScreen)
        await pilot.press("h")
        await pilot.pause()
        assert isinstance(app.screen, SessionsScreen)


async def test_vim_navigation_l_selects(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        await pilot.press("l")
        await pilot.pause()
        assert isinstance(app.screen, MessagesScreen)


async def test_streaming_record_assembles_response(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        # First row is req-002 (streaming)
        await pilot.press("enter")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, MessagesScreen)
        # 1 request message + 1 assembled assistant response = 2
        table = screen.query_one("DataTable")
        assert table.row_count == 2
        assert table_has_text(screen, "chunk")


# ---------------------------------------------------------------------------
# Anthropic format tests
# ---------------------------------------------------------------------------

async def test_anthropic_response_extracted(anthropic_storage):
    app = MalcolmTUI(db_path=anthropic_storage._db_path)
    async with app.run_test() as pilot:
        # req-anth-003 is newest (first row), req-anth-001 is oldest (third row)
        # Navigate to req-anth-001 (non-streaming Anthropic)
        await pilot.press("j")
        await pilot.press("j")
        await pilot.press("enter")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, MessagesScreen)
        assert table_has_text(screen, "Hi from Claude!")


async def test_anthropic_streaming_assembled(anthropic_storage):
    app = MalcolmTUI(db_path=anthropic_storage._db_path)
    async with app.run_test() as pilot:
        # req-anth-002 is second row (streaming Anthropic)
        await pilot.press("j")
        await pilot.press("enter")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, MessagesScreen)
        assert table_has_text(screen, "Hello world!")


async def test_tool_call_shows_tool_name(anthropic_storage):
    app = MalcolmTUI(db_path=anthropic_storage._db_path)
    async with app.run_test() as pilot:
        # req-anth-003 is first row (tool call)
        await pilot.press("enter")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, MessagesScreen)
        assert table_has_text(screen, "tool: read_file")


# ---------------------------------------------------------------------------
# System-reminder filtering
# ---------------------------------------------------------------------------

async def test_system_reminder_filtered_from_preview(anthropic_storage):
    app = MalcolmTUI(db_path=anthropic_storage._db_path)
    async with app.run_test() as pilot:
        # req-anth-001 has user message with system-reminder + "hello"
        await pilot.press("j")
        await pilot.press("j")
        await pilot.press("enter")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, MessagesScreen)
        # Should show "hello" but not "system-reminder"
        assert table_has_text(screen, "hello")
        assert not table_has_text(screen, "hook output")


# ---------------------------------------------------------------------------
# Role styles
# ---------------------------------------------------------------------------

async def test_messages_have_role_styles(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        await pilot.press("enter")  # enter session
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, MessagesScreen)
        table = screen.query_one("DataTable")
        # Check that role cells are Rich Text objects (styled)
        col_keys = list(table.columns.keys())
        role_col = col_keys[1]  # "#", "Role", "Content"
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

        # Insert a record while TUI is running
        record = RequestRecord(
            id="req-new",
            model="gpt-4o",
            request_body={
                "model": "gpt-4o",
                "messages": [],
                "metadata": {"user_id": '{"session_id": "sess-new"}'},
            },
            status_code=200,
            duration_ms=100.0,
        )
        await storage.save(record)

        await pilot.press("r")
        await pilot.pause()
        table = app.screen.query_one("DataTable")
        assert table.row_count == 1  # the new record
        assert table_has_text(app.screen, "gpt-4o")


# ---------------------------------------------------------------------------
# Header subtitle updates on back navigation
# ---------------------------------------------------------------------------

async def test_subtitle_updates_on_back(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        assert app.sub_title == "Sessions"
        await pilot.press("enter")
        await pilot.pause()
        assert "gpt-4o" in app.sub_title
        await pilot.press("escape")
        await pilot.pause()
        assert app.sub_title == "Sessions"


# ---------------------------------------------------------------------------
# Detail screen
# ---------------------------------------------------------------------------

async def test_detail_shows_json(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
        await pilot.press("enter")  # enter session
        await pilot.pause()
        await pilot.press("enter")  # first message
        await pilot.pause()
        assert isinstance(app.screen, DetailScreen)
        assert "user" in app.sub_title.lower()


async def test_detail_back_to_messages(populated_storage):
    app = MalcolmTUI(db_path=populated_storage._db_path)
    async with app.run_test() as pilot:
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
