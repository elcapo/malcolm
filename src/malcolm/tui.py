"""Malcolm TUI — terminal log viewer."""

from __future__ import annotations

import json

from rich.syntax import Syntax
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import DataTable, Header, RichLog, Static

from malcolm.storage import Storage


def _extract_assistant_message(resp_body: dict) -> dict | None:
    """Extract the assistant message from a response body (OpenAI or Anthropic format)."""
    if not isinstance(resp_body, dict):
        return None
    # OpenAI format: choices[].message
    if "choices" in resp_body:
        for choice in resp_body["choices"]:
            msg = choice.get("message") or choice.get("delta")
            if msg and (msg.get("content") is not None or msg.get("tool_calls")):
                return msg
    # Anthropic format: role + content[]
    if resp_body.get("role") == "assistant" and "content" in resp_body:
        return {"role": "assistant", "content": resp_body["content"]}
    return None


def _assemble_anthropic_chunks(chunks: list[dict]) -> dict | None:
    """Assemble Anthropic SSE chunks into a single assistant message."""
    text_parts: list[str] = []
    for chunk in chunks:
        # Anthropic content_block_delta with text_delta
        delta = chunk.get("delta", {})
        if delta.get("type") == "text_delta" and delta.get("text"):
            text_parts.append(delta["text"])
        # OpenAI format fallback
        for choice in chunk.get("choices", []):
            d = choice.get("delta", {})
            if d.get("content"):
                text_parts.append(d["content"])
    if text_parts:
        return {"role": "assistant", "content": "".join(text_parts)}
    return None


_HINT_NAV = "↑/k ↓/j navigate · →/l open · r reload · p theme · q quit"
_HINT_NAV_BACK = "↑/k ↓/j navigate · →/l open · ←/h back · r reload · p theme · q quit"
_HINT_DETAIL = "↑/k ↓/j scroll · ←/h back · w wrap · r reload · p theme · q quit"


class HintBar(Static):
    """Bottom bar showing keyboard hints."""

    DEFAULT_CSS = """
    HintBar {
        dock: bottom;
        height: 1;
        background: $panel;
        color: $text-muted;
        text-align: center;
    }
    """


class VimDataTable(DataTable):
    """DataTable with vim-style navigation."""

    BINDINGS = [
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("l", "select_cursor", "Select", show=False),
        Binding("r", "reload_screen", "Reload", show=False),
    ]

    def action_cursor_left(self) -> None:
        """Left arrow goes back instead of horizontal scroll."""
        if hasattr(self.screen, "action_back"):
            self.screen.action_back()

    def action_cursor_right(self) -> None:
        """Right arrow selects instead of horizontal scroll."""
        self.action_select_cursor()


class SessionsScreen(Screen):
    """List of sessions."""

    def compose(self) -> ComposeResult:
        yield Header()
        yield VimDataTable(cursor_type="row")
        yield HintBar(_HINT_NAV)

    async def on_mount(self) -> None:
        self.app.sub_title = "Sessions"
        await self._load_sessions()

    async def _load_sessions(self) -> None:
        await self.app.storage.refresh()
        table = self.query_one(VimDataTable)
        table.clear(columns=True)
        table.add_columns("Time", "Model", "Messages", "Session")
        sessions = await self.app.storage.list_sessions(100)
        self._sessions = sessions
        for s in sessions:
            table.add_row(
                s["timestamp"][:19],
                s.get("model") or "-",
                str(s["user_messages"]),
                s["session_id"][:12] + "…",
                key=s["last_request_id"],
            )
        if not sessions:
            table.add_row("-", "No sessions", "-", "-")

    async def action_reload(self) -> None:
        await self._load_sessions()

    def on_screen_resume(self) -> None:
        self.app.sub_title = "Sessions"

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if not self._sessions:
            return
        self.app.push_screen(MessagesScreen(event.row_key.value))


class MessagesScreen(Screen):
    """Messages within a single request."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("h", "back", "Back", show=False),
    ]

    def __init__(self, record_id: str) -> None:
        super().__init__()
        self.record_id = record_id
        self._messages: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield VimDataTable(cursor_type="row")
        yield HintBar(_HINT_NAV_BACK)

    async def on_mount(self) -> None:
        await self._load_messages()

    async def _load_messages(self) -> None:
        await self.app.storage.refresh()
        record = await self.app.storage.get(self.record_id)
        if not record:
            self.app.pop_screen()
            return

        model = record.get("model") or "?"
        ts = (record.get("timestamp") or "")[:19]
        self._sub_title = f"{model} — {ts}"
        self.app.sub_title = self._sub_title

        table = self.query_one(VimDataTable)
        table.clear(columns=True)
        table.add_columns("#", "Role", "Content")

        # Collect request messages
        req_body = record.get("request_body") or {}
        messages = list(req_body.get("messages") or [])

        # Add assistant response
        resp_body = record.get("response_body") or {}
        resp_msg = _extract_assistant_message(resp_body)
        if resp_msg:
            messages.append(resp_msg)
        elif record.get("response_chunks"):
            chunks = record["response_chunks"]
            assembled = _assemble_anthropic_chunks(chunks)
            if assembled:
                messages.append(assembled)
            else:
                messages.append({
                    "role": "assistant",
                    "content": f"[streaming — {len(chunks)} chunks]",
                    "_chunks": chunks,
                })

        self._messages = messages

        for i, msg in enumerate(messages):
            content = msg.get("content") or ""
            if isinstance(content, list):
                # Take the last non-system-reminder text block
                last_text = ""
                for item in reversed(content):
                    if isinstance(item, dict):
                        text = item.get("text", "")
                        if text and not text.lstrip().startswith("<system-reminder>"):
                            last_text = text
                            break
                        elif item.get("type") and item["type"] != "text":
                            last_text = f"[{item['type']}]"
                            break
                    elif item:
                        last_text = str(item)
                        break
                content = last_text
            if not content and msg.get("tool_calls"):
                names = [tc.get("function", {}).get("name", "?") for tc in msg["tool_calls"]]
                content = f"[tool: {', '.join(names)}]"
            preview = (content or "—")[:100].replace("\n", " ")
            role = msg.get("role") or "?"
            if role == "user":
                style = "bold"
            elif role == "assistant":
                style = "italic"
            elif role == "system":
                style = "dim"
            else:
                style = ""
            table.add_row(
                str(i + 1),
                Text(role, style=style),
                Text(preview, style=style),
                key=str(i),
            )

    async def on_key(self, event) -> None:
        if event.key == "r":
            event.prevent_default()
            event.stop()
            while not isinstance(self.app.screen, SessionsScreen):
                self.app.pop_screen()
            await self.app.screen.action_reload()

    def on_screen_resume(self) -> None:
        if hasattr(self, "_sub_title"):
            self.app.sub_title = self._sub_title

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        idx = int(event.row_key.value)
        self.app.push_screen(DetailScreen(self._messages[idx]))

    def action_back(self) -> None:
        self.app.pop_screen()


class DetailScreen(Screen):
    """Full JSON detail of a single message."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("h", "back", "Back", show=False),
        Binding("left", "back", "Back", show=False, priority=True),
        Binding("j", "scroll_down", "Down", show=False),
        Binding("k", "scroll_up", "Up", show=False),
        Binding("w", "toggle_wrap", "Wrap", show=False),
    ]

    def __init__(self, message: dict) -> None:
        super().__init__()
        self._message = message
        self._wrap = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(highlight=True, wrap=False)
        yield HintBar(_HINT_DETAIL)

    def on_mount(self) -> None:
        self._render_message()

    def _render_message(self) -> None:
        role = self._message.get("role", "message")
        self.app.sub_title = f"Detail — {role}"
        log = self.query_one(RichLog)
        log.clear()
        formatted = json.dumps(self._message, indent=2, ensure_ascii=False)
        log.write(Syntax(formatted, "json", theme="monokai", word_wrap=self._wrap))

    def action_toggle_wrap(self) -> None:
        self._wrap = not self._wrap
        log = self.query_one(RichLog)
        log.wrap = self._wrap
        self._render_message()

    def action_back(self) -> None:
        self.app.pop_screen()

    async def on_key(self, event) -> None:
        if event.key == "r":
            event.prevent_default()
            event.stop()
            # Pop back to sessions and reload
            while not isinstance(self.app.screen, SessionsScreen):
                self.app.pop_screen()
            await self.app.screen.action_reload()

    def action_scroll_down(self) -> None:
        self.query_one(RichLog).scroll_down()

    def action_scroll_up(self) -> None:
        self.query_one(RichLog).scroll_up()


class MalcolmTUI(App):
    """Malcolm TUI — LLM request log viewer."""

    TITLE = "Malcolm"
    BINDINGS = [
        Binding("p", "toggle_dark", "Theme", show=False),
        Binding("r", "reload_screen", "Reload", show=False),
        Binding("q", "quit", "Quit"),
    ]
    CSS = """
    DataTable { height: 1fr; scrollbar-size: 0 0; }
    RichLog { height: 1fr; scrollbar-size: 0 0; }
    """

    def __init__(self, db_path: str) -> None:
        super().__init__()
        self._db_path = db_path
        self.storage: Storage

    async def action_reload_screen(self) -> None:
        screen = self.screen
        if hasattr(screen, "action_reload"):
            await screen.action_reload()

    async def on_mount(self) -> None:
        self.storage = Storage(self._db_path)
        await self.storage.init()
        self.push_screen(SessionsScreen())

    async def on_unmount(self) -> None:
        if hasattr(self, "storage"):
            await self.storage.close()


def run_tui(db_path: str | None = None) -> None:
    if db_path is None:
        db_path = "malcolm.db"
    app = MalcolmTUI(db_path=db_path)
    app.run()
