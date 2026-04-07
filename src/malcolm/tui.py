"""Malcolm TUI — terminal log viewer."""

from __future__ import annotations

import json

from rich.syntax import Syntax
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import DataTable, Header, RichLog, Static

from malcolm.formats import group_records, parse_record
from malcolm.models import Conversation, Message, SessionGroup
from malcolm.storage import Storage

_PAGE_SIZE = 50
_VIEW_RAW = "raw"


def _record_for_view(record: dict, view: str) -> dict:
    """Return a virtual record using the given transform's bodies, or raw."""
    if view == _VIEW_RAW:
        return record
    for t in record.get("transforms", []):
        if t["transform_type"] == view:
            virtual = dict(record)
            if t.get("request_body") is not None:
                virtual["request_body"] = t["request_body"]
            if t.get("response_body") is not None:
                virtual["response_body"] = t["response_body"]
            if t.get("response_chunks") is not None:
                virtual["response_chunks"] = t["response_chunks"]
            return virtual
    return record


def _available_views(record: dict) -> list[str]:
    """Return the list of available views for a record."""
    views = [_VIEW_RAW]
    for t in record.get("transforms", []):
        views.append(t["transform_type"])
    return views

_HINT_GROUPS = "↑/k ↓/j navigate · →/l open · n next page · N prev page · r reload · p theme · q quit"
_HINT_NAV_BACK = "↑/k ↓/j navigate · →/l open · ←/h back · t view · r reload · p theme · q quit"
_HINT_DETAIL = "↑/k ↓/j scroll · ←/h back · t view · w wrap · r reload · p theme · q quit"


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
    ]

    def action_cursor_left(self) -> None:
        """Left arrow goes back instead of horizontal scroll."""
        if hasattr(self.screen, "action_back"):
            self.screen.action_back()

    def action_cursor_right(self) -> None:
        """Right arrow selects instead of horizontal scroll."""
        self.action_select_cursor()


class GroupsScreen(Screen):
    """List of session groups (first screen)."""

    BINDINGS = [
        Binding("n", "next_page", "Next page", show=False),
        Binding("N", "prev_page", "Prev page", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._groups: list[SessionGroup] = []
        self._page = 0
        self._pages: list[list[SessionGroup]] = []
        self._cursor: str | None = None  # timestamp cursor for pagination
        self._has_more = True

    def compose(self) -> ComposeResult:
        yield Header()
        yield VimDataTable(cursor_type="row")
        yield HintBar(_HINT_GROUPS)

    async def on_mount(self) -> None:
        self.app.sub_title = "Sessions"
        await self._load_page(reset=True)

    async def _load_page(self, reset: bool = False) -> None:
        await self.app.storage.refresh()
        if reset:
            self._pages = []
            self._page = 0
            self._cursor = None
            self._has_more = True

        # Only fetch if we need a new page
        if self._page >= len(self._pages) and self._has_more:
            records = await self.app.storage.list_page_full(
                page_size=_PAGE_SIZE, before=self._cursor,
            )
            if records:
                groups = group_records(records)
                self._pages.append(groups)
                self._cursor = records[-1].get("timestamp")
                self._has_more = len(records) == _PAGE_SIZE
            else:
                self._has_more = False
                if not self._pages:
                    self._pages.append([])

        self._groups = self._pages[self._page] if self._page < len(self._pages) else []
        self._render_table()

    def _render_table(self) -> None:
        table = self.query_one(VimDataTable)
        table.clear(columns=True)
        table.add_columns("Model", "First message", "Last message", "Requests")

        for g in self._groups:
            table.add_row(
                Text(g.model or "-", style="bold cyan"),
                g.earliest_timestamp[:19],
                g.latest_timestamp[:19],
                Text(str(g.request_count), style="", justify="right"),
                key=g.session_id,
            )

        if not self._groups:
            table.add_row("-", "-", "No sessions", "-")

        page_info = f"page {self._page + 1}"
        if not self._has_more and self._page >= len(self._pages) - 1:
            page_info += " (last)"
        self.app.sub_title = f"Sessions — {page_info}"

    async def action_next_page(self) -> None:
        if self._has_more or self._page < len(self._pages) - 1:
            self._page += 1
            await self._load_page()

    async def action_prev_page(self) -> None:
        if self._page > 0:
            self._page -= 1
            self._groups = self._pages[self._page]
            self._render_table()

    async def action_reload(self) -> None:
        await self._load_page(reset=True)

    def on_screen_resume(self) -> None:
        page_info = f"page {self._page + 1}"
        self.app.sub_title = f"Sessions — {page_info}"

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if not self._groups:
            return
        sid = event.row_key.value
        for g in self._groups:
            if g.session_id == sid:
                self.app.push_screen(RequestsScreen(g))
                return


class RequestsScreen(Screen):
    """List of requests within a session group."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("h", "back", "Back", show=False),
    ]

    def __init__(self, group: SessionGroup) -> None:
        super().__init__()
        self._group = group
        self._records: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield VimDataTable(cursor_type="row")
        yield HintBar(_HINT_NAV_BACK)

    async def on_mount(self) -> None:
        await self._load_requests()

    async def _load_requests(self) -> None:
        await self.app.storage.refresh()
        records = []
        for rid in self._group.record_ids:
            record = await self.app.storage.get(rid)
            if record:
                records.append(record)
        self._records = records

        model = self._group.model or "?"
        self._sub_title = f"{model} — {self._group.request_count} requests"
        self.app.sub_title = self._sub_title

        table = self.query_one(VimDataTable)
        table.clear(columns=True)
        table.add_columns("Model", "Time", "Status", "Duration", "Stream")

        for r in records:
            table.add_row(
                Text(r.get("model") or "-", style="bold cyan"),
                (r.get("timestamp") or "")[:19],
                _format_status(r.get("status_code")),
                _format_duration(r.get("duration_ms")),
                "Yes" if r.get("stream") else "No",
                key=r["id"],
            )

        if not records:
            table.add_row("-", "-", "-", "-", "-")

    async def action_reload(self) -> None:
        await self._load_requests()

    def on_screen_resume(self) -> None:
        if hasattr(self, "_sub_title"):
            self.app.sub_title = self._sub_title

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if not self._records:
            return
        self.app.push_screen(MessagesScreen(event.row_key.value))

    def action_back(self) -> None:
        self.app.pop_screen()


class MessagesScreen(Screen):
    """Messages within a single request."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("h", "back", "Back", show=False),
        Binding("t", "toggle_view", "Toggle view", show=False),
    ]

    def __init__(self, record_id: str) -> None:
        super().__init__()
        self.record_id = record_id
        self._conversation: Conversation | None = None
        self._record: dict | None = None
        self._views: list[str] = [_VIEW_RAW]
        self._view_idx: int = 0

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

        self._record = record
        self._views = _available_views(record)
        self._view_idx = 0
        self._render_for_view()

    def _view_label(self) -> str:
        if len(self._views) <= 1:
            return ""
        n = self._view_idx + 1
        total = len(self._views)
        name = self._views[self._view_idx]
        return f" — view {n}/{total} ({name})"

    def _render_for_view(self) -> None:
        view = self._views[self._view_idx]
        view_record = _record_for_view(self._record, view)
        conversation = parse_record(view_record)
        self._conversation = conversation

        model = conversation.model or "?"
        ts = conversation.timestamp[:19]
        self._sub_title = f"{model} — {ts}{self._view_label()}"
        self.app.sub_title = self._sub_title

        table = self.query_one(VimDataTable)
        table.clear(columns=True)
        table.add_columns("#", "Role", "Content")

        for i, msg in enumerate(conversation.messages):
            preview = _message_preview(msg)
            role = msg.role
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

    def action_toggle_view(self) -> None:
        if self._record is None or len(self._views) <= 1:
            return
        self._view_idx = (self._view_idx + 1) % len(self._views)
        self._render_for_view()

    async def action_reload(self) -> None:
        await self._load_messages()

    def on_screen_resume(self) -> None:
        if hasattr(self, "_sub_title"):
            self.app.sub_title = self._sub_title

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if not self._conversation:
            return
        idx = int(event.row_key.value)
        self.app.push_screen(DetailScreen(
            self._conversation.messages[idx],
            record=self._record,
            views=self._views,
            view_idx=self._view_idx,
        ))

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
        Binding("t", "toggle_view", "Toggle view", show=False),
    ]

    def __init__(
        self,
        message: Message,
        record: dict | None = None,
        views: list[str] | None = None,
        view_idx: int = 0,
    ) -> None:
        super().__init__()
        self._message = message
        self._msg_idx: int | None = None
        self._record = record
        self._views = views or [_VIEW_RAW]
        self._view_idx = view_idx
        self._wrap = True

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(highlight=True, wrap=True)
        yield HintBar(_HINT_DETAIL)

    def on_mount(self) -> None:
        # Find the message index in the current conversation
        if self._record is not None:
            view = self._views[self._view_idx]
            conv = parse_record(_record_for_view(self._record, view))
            for i, m in enumerate(conv.messages):
                if m.raw == self._message.raw:
                    self._msg_idx = i
                    break
        self._render_message()

    def _view_label(self) -> str:
        if len(self._views) <= 1:
            return ""
        n = self._view_idx + 1
        total = len(self._views)
        name = self._views[self._view_idx]
        return f" — view {n}/{total} ({name})"

    def _render_message(self) -> None:
        self.app.sub_title = f"Detail — {self._message.role}{self._view_label()}"
        log = self.query_one(RichLog)
        log.clear()
        formatted = json.dumps(self._message.raw, indent=2, ensure_ascii=False)
        log.write(Syntax(formatted, "json", theme="monokai", word_wrap=self._wrap))

    def action_toggle_view(self) -> None:
        if self._record is None or len(self._views) <= 1:
            return
        self._view_idx = (self._view_idx + 1) % len(self._views)
        view = self._views[self._view_idx]
        conv = parse_record(_record_for_view(self._record, view))
        if self._msg_idx is not None and self._msg_idx < len(conv.messages):
            self._message = conv.messages[self._msg_idx]
        self._render_message()

    def action_toggle_wrap(self) -> None:
        self._wrap = not self._wrap
        log = self.query_one(RichLog)
        log.wrap = self._wrap
        self._render_message()

    def action_back(self) -> None:
        self.app.pop_screen()

    async def action_reload(self) -> None:
        self._render_message()

    def action_scroll_down(self) -> None:
        self.query_one(RichLog).scroll_down()

    def action_scroll_up(self) -> None:
        self.query_one(RichLog).scroll_up()


def _format_status(status: int | None) -> Text:
    """Format an HTTP status code with a color based on its class."""
    if status is None:
        return Text("-")
    if 200 <= status < 300:
        style = "green"
    elif 300 <= status < 400:
        style = "cyan"
    elif 400 <= status < 500:
        style = "red"
    elif 500 <= status < 600:
        style = "bold red"
    else:
        style = ""
    return Text(str(status), style=style)


def _format_duration(duration_ms: float | None) -> Text:
    """Format a duration in ms with thousands separator, right-aligned."""
    if duration_ms is None:
        return Text("-", justify="right")
    return Text(f"{duration_ms:,.0f} ms", justify="right")


def _message_preview(msg: Message) -> str:
    """Build a short preview string for a message."""
    if msg.text:
        content = msg.raw.get("content") if msg.raw else None
        if isinstance(content, list):
            text_blocks = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            if text_blocks:
                return text_blocks[-1][:100].replace("\n", " ")
        return msg.text[:100].replace("\n", " ")
    if msg.tool_calls:
        names = [tc.name for tc in msg.tool_calls]
        return f"[tool: {', '.join(names)}]"
    if msg.tool_result:
        return msg.tool_result[:100].replace("\n", " ")
    return "—"


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
        self.push_screen(GroupsScreen())

    async def on_unmount(self) -> None:
        if hasattr(self, "storage"):
            await self.storage.close()


def run_tui(db_path: str | None = None) -> None:
    if db_path is None:
        db_path = "malcolm.db"
    app = MalcolmTUI(db_path=db_path)
    app.run()
