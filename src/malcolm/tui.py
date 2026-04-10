"""Malcolm TUI — agnostic terminal log viewer.

Displays a flat list of proxy requests with optional annotation-driven
columns and detail views.  The TUI never imports domain-specific code
(formats, models); all enrichment comes from transform annotations stored
in the database.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import OrderedDict

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.events import Key
from textual.screen import Screen
from textual.timer import Timer
from textual.widgets import DataTable, Header, Static, TextArea

from malcolm.storage import Storage

_PAGE_SIZE = 50
_FOLLOW_INTERVAL = 1.0

_HINT_LIST = (
    "\u2191\u2193 Navigate \u00b7 Enter Open \u00b7 "
    "[bold]N[/bold] Next \u00b7 [bold]P[/bold] Prev \u00b7 "
    "[bold]F[/bold] Follow \u00b7 [bold]R[/bold] Reload \u00b7 "
    "[bold]T[/bold] Theme \u00b7 [bold]Q[/bold] Quit"
)
_HINT_ANNOTATIONS = (
    "\u2191\u2193 Navigate \u00b7 Enter Open \u00b7 Esc Back \u00b7 "
    "[bold]R[/bold] Reload \u00b7 [bold]T[/bold] Theme \u00b7 [bold]Q[/bold] Quit"
)
_HINT_CONTENT = (
    "\u2191\u2193 Scroll \u00b7 Esc Back \u00b7 "
    "[bold]C[/bold] Copy \u00b7 [bold]W[/bold] Wrap \u00b7 "
    "[bold]T[/bold] Theme \u00b7 [bold]Q[/bold] Quit"
)


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


class ReadOnlyTextArea(TextArea):
    """TextArea that lets printable keys (p, q, w, …) bubble up to the screen/app."""

    def _on_key(self, event: Key) -> None:
        if self.read_only and event.character and event.character.isprintable():
            return
        super()._on_key(event)


# ── Screen 1: Request list ──────────────────────────────────────────


class RequestListScreen(Screen):
    """Flat list of all proxy requests with dynamic badge columns."""

    BINDINGS = [
        Binding("n", "next_page", "Next page", show=False),
        Binding("p", "prev_page", "Prev page", show=False),
        Binding("f", "toggle_follow", "Follow", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._records: list[dict] = []
        self._page = 0
        self._pages: list[list[dict]] = []
        self._cursor: str | None = None
        self._has_more = True
        self._following = False
        self._follow_timer: Timer | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(cursor_type="row")
        yield HintBar(_HINT_LIST)

    async def on_mount(self) -> None:
        self.app.sub_title = "Requests"
        await self._load_page(reset=True)

    async def _load_page(self, reset: bool = False) -> None:
        await self.app.storage.refresh()
        if reset:
            self._pages = []
            self._page = 0
            self._cursor = None
            self._has_more = True

        if self._page >= len(self._pages) and self._has_more:
            records = await self.app.storage.list_page_with_badges(
                page_size=_PAGE_SIZE, before=self._cursor,
            )
            if records:
                self._pages.append(records)
                self._cursor = records[-1].get("timestamp")
                self._has_more = len(records) == _PAGE_SIZE
            else:
                self._has_more = False
                if not self._pages:
                    self._pages.append([])

        self._records = (
            self._pages[self._page] if self._page < len(self._pages) else []
        )
        self._render_table()

    def _render_table(self) -> None:
        table = self.query_one(DataTable)
        table.clear(columns=True)

        # Discover dynamic badge columns from current page
        badge_keys: OrderedDict[str, None] = OrderedDict()
        for r in self._records:
            for k in r.get("badges", {}):
                badge_keys[k] = None

        # Fixed columns + dynamic badge columns
        columns = ["ID", "Timestamp", "Status", "Duration"]
        for bk in badge_keys:
            columns.append(bk.replace("_", " ").title())
        table.add_columns(*columns)

        for r in self._records:
            row: list[str | Text] = [
                Text(r["id"][:8], style="dim"),
                (r.get("timestamp") or "")[:19],
                _format_status(r.get("status_code")),
                _format_duration(r.get("duration_ms")),
            ]
            badges = r.get("badges", {})
            for bk in badge_keys:
                val = badges.get(bk, "")
                row.append(Text(val, style="bold cyan") if val else Text("-"))
            table.add_row(*row, key=r["id"])

        if not self._records:
            empty_row: list[str | Text] = ["-", "-", "-", "No requests"]
            for _ in badge_keys:
                empty_row.append("-")
            table.add_row(*empty_row)

        self._update_subtitle()

    def _update_subtitle(self) -> None:
        page_info = f"page {self._page + 1}"
        if not self._has_more and self._page >= len(self._pages) - 1:
            page_info += " (last)"
        follow_tag = " [FOLLOW]" if self._following else ""
        self.app.sub_title = f"Requests \u2014 {page_info}{follow_tag}"

    async def action_next_page(self) -> None:
        if self._has_more or self._page < len(self._pages) - 1:
            self._page += 1
            await self._load_page()

    async def action_prev_page(self) -> None:
        if self._page > 0:
            self._page -= 1
            self._records = self._pages[self._page]
            self._render_table()

    async def action_reload(self) -> None:
        await self._load_page(reset=True)

    def action_toggle_follow(self) -> None:
        self._following = not self._following
        if self._following:
            self._follow_timer = self.set_interval(
                _FOLLOW_INTERVAL, self._follow_tick,
            )
        elif self._follow_timer is not None:
            self._follow_timer.stop()
            self._follow_timer = None
        self._update_subtitle()

    async def _follow_tick(self) -> None:
        await self._load_page(reset=True)

    def on_screen_resume(self) -> None:
        self._update_subtitle()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if not self._records:
            return
        record_id = event.row_key.value
        self.app.push_screen(AnnotationsScreen(record_id))


# ── Screen 2: Annotations list ──────────────────────────────────────


class AnnotationsScreen(Screen):
    """Shows annotations for a request as navigable rows, grouped by source."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
    ]

    def __init__(self, record_id: str) -> None:
        super().__init__()
        self.record_id = record_id
        self._record: dict | None = None
        self._annotations: list[dict] = []
        self._rows: list[dict] = []  # unified list for row selection

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(cursor_type="row")
        yield HintBar(_HINT_ANNOTATIONS)

    async def on_mount(self) -> None:
        await self._load()

    async def _load(self) -> None:
        await self.app.storage.refresh()
        record = await self.app.storage.get(self.record_id)
        if not record:
            self.app.pop_screen()
            return

        self._record = record
        self._annotations = record.get("annotations", [])
        self._render_table()

    def _render_table(self) -> None:
        table = self.query_one(DataTable)
        table.clear(columns=True)
        table.add_columns("Source", "Category", "Key", "Preview")

        self._rows = []
        row_idx = 0

        # Group annotations by source, then by category
        by_source: OrderedDict[str, list[dict]] = OrderedDict()
        for a in self._annotations:
            src = a.get("source") or "other"
            by_source.setdefault(src, []).append(a)

        for source, anns in by_source.items():
            for a in anns:
                display = a.get("display", "kv")
                value = a["value"]
                if display in ("kv", "badge"):
                    preview = value[:80]
                elif display == "text":
                    preview = value[:80].replace("\n", " ")
                    if len(value) > 80:
                        preview += "\u2026"
                elif display == "json":
                    preview = value[:80].replace("\n", " ")
                else:
                    preview = value[:80]

                self._rows.append({
                    "type": "annotation",
                    "annotation": a,
                    "source": source,
                })
                table.add_row(
                    Text(source, style="dim"),
                    Text(a.get("category") or "-", style="italic"),
                    Text(a["key"], style="bold"),
                    preview,
                    key=str(row_idx),
                )
                row_idx += 1

        # Add raw JSON entries
        for label, body_key in [("request", "request_body"), ("response", "response_body")]:
            body = self._record.get(body_key) if self._record else None
            if body is not None:
                self._rows.append({"type": "raw", "label": label, "body_key": body_key})
                table.add_row(
                    Text("\u2500\u2500\u2500", style="dim"),
                    Text("\u2500\u2500\u2500", style="dim"),
                    Text(f"[raw {label}]", style="bold cyan"),
                    Text("full JSON body", style="dim"),
                    key=str(row_idx),
                )
                row_idx += 1

        # Add transform body entries
        if self._record:
            for t in self._record.get("transforms", []):
                tname = t["transform_type"]
                for side in ("request", "response"):
                    body_key = f"{side}_body"
                    if t.get(body_key) is not None:
                        self._rows.append({
                            "type": "transform",
                            "transform": t,
                            "label": f"{tname}:{side}",
                            "body_key": body_key,
                        })
                        table.add_row(
                            Text("\u2500\u2500\u2500", style="dim"),
                            Text("\u2500\u2500\u2500", style="dim"),
                            Text(f"[{tname}:{side}]", style="bold cyan"),
                            Text("transformed JSON body", style="dim"),
                            key=str(row_idx),
                        )
                        row_idx += 1

        if not self._rows:
            table.add_row("-", "-", "-", "No annotations")

        self._sub_title = f"Detail \u2014 {self.record_id[:8]}"
        self.app.sub_title = self._sub_title

    def on_screen_resume(self) -> None:
        if hasattr(self, "_sub_title"):
            self.app.sub_title = self._sub_title

    async def action_reload(self) -> None:
        await self._load()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if not self._rows:
            return
        idx = int(event.row_key.value)
        if idx >= len(self._rows):
            return
        entry = self._rows[idx]

        if entry["type"] == "annotation":
            a = entry["annotation"]
            self.app.push_screen(ContentScreen(
                title=f"{a['key']}",
                content=a["value"],
                syntax="json" if a.get("display") == "json" else None,
            ))
        elif entry["type"] == "raw":
            body = self._record.get(entry["body_key"]) if self._record else None
            if body is not None:
                self.app.push_screen(ContentScreen(
                    title=f"raw {entry['label']}",
                    content=json.dumps(body, indent=2, ensure_ascii=False),
                    syntax="json",
                ))
        elif entry["type"] == "transform":
            body = entry["transform"].get(entry["body_key"])
            if body is not None:
                self.app.push_screen(ContentScreen(
                    title=entry["label"],
                    content=json.dumps(body, indent=2, ensure_ascii=False),
                    syntax="json",
                ))

    def action_back(self) -> None:
        self.app.pop_screen()


# ── Screen 3: Content view ──────────────────────────────────────────


class ContentScreen(Screen):
    """Full-screen view of a single annotation value or raw JSON body.

    Uses ``TextArea`` in read-only mode for fast virtual rendering
    (only visible lines are painted) with tree-sitter syntax highlighting.
    """

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("left", "back", "Back", show=False, priority=True),
        Binding("w", "toggle_wrap", "Wrap", show=False),
        Binding("c", "copy", "Copy", show=False),
    ]

    def __init__(
        self,
        title: str,
        content: str,
        syntax: str | None = None,
    ) -> None:
        super().__init__()
        self._title = title
        self._content = content
        self._syntax = syntax
        self._wrap = True

    def compose(self) -> ComposeResult:
        yield Header()
        lang = self._syntax or "markdown"
        yield ReadOnlyTextArea(
            self._content,
            language=lang,
            read_only=True,
            soft_wrap=self._wrap,
            show_line_numbers=True,
        )
        yield HintBar(_HINT_CONTENT)

    def on_mount(self) -> None:
        self.app.sub_title = f"Content \u2014 {self._title}"
        ta_theme = MalcolmTUI._TEXTAREA_THEMES.get(self.app.theme, "monokai")
        self.query_one(ReadOnlyTextArea).theme = ta_theme

    def action_toggle_wrap(self) -> None:
        self._wrap = not self._wrap
        ta = self.query_one(ReadOnlyTextArea)
        ta.soft_wrap = self._wrap

    def action_copy(self) -> None:
        ta = self.query_one(ReadOnlyTextArea)
        text = ta.selected_text or self._content
        _copy_to_clipboard(text)
        self.app.notify("Copied to clipboard")

    def action_back(self) -> None:
        self.app.pop_screen()


# ── Helpers ──────────────────────────────────────────────────────────


def _copy_to_clipboard(text: str) -> None:
    """Copy text to the system clipboard using platform tools."""
    for cmd in ("wl-copy", "xclip", "pbcopy"):
        path = shutil.which(cmd)
        if path is None:
            continue
        args = [path] if cmd != "xclip" else [path, "-selection", "clipboard"]
        subprocess.run(args, input=text.encode(), check=True)
        return
    raise RuntimeError("No clipboard tool found (install wl-copy, xclip, or pbcopy)")


def _format_status(status: int | None) -> Text:
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
    if duration_ms is None:
        return Text("-", justify="right")
    return Text(f"{duration_ms:,.0f} ms", justify="right")


# ── App ──────────────────────────────────────────────────────────────


class MalcolmTUI(App):
    """Malcolm TUI — LLM request log viewer."""

    TITLE = "Malcolm"
    BINDINGS = [
        Binding("t", "toggle_dark", "Theme", show=False),
        Binding("r", "reload_screen", "Reload", show=False),
        Binding("q", "quit", "Quit"),
    ]
    CSS = """
    DataTable { height: 1fr; scrollbar-size: 0 0; }
    TextArea { height: 1fr; }
    """

    def __init__(self, db_path: str) -> None:
        super().__init__()
        self._db_path = db_path
        self.storage: Storage

    _TEXTAREA_THEMES = {
        "textual-dark": "monokai",
        "textual-light": "github_light",
    }

    def _sync_textarea_theme(self) -> None:
        ta_theme = self._TEXTAREA_THEMES.get(self.theme, "monokai")
        for ta in self.screen.query(ReadOnlyTextArea):
            ta.theme = ta_theme

    def action_toggle_dark(self) -> None:
        super().action_toggle_dark()
        self._sync_textarea_theme()

    async def action_reload_screen(self) -> None:
        screen = self.screen
        if hasattr(screen, "action_reload"):
            await screen.action_reload()

    async def on_mount(self) -> None:
        self.storage = Storage(self._db_path)
        await self.storage.init()
        self.push_screen(RequestListScreen())

    async def on_unmount(self) -> None:
        if hasattr(self, "storage"):
            await self.storage.close()


def run_tui(db_path: str | None = None) -> None:
    if db_path is None:
        db_path = "malcolm.db"
    app = MalcolmTUI(db_path=db_path)
    app.run()
