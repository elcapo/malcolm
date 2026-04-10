"""Transform protocol — the interface every transform must implement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Annotation:
    """A structured observation produced by a transform about a request.

    Annotations let transforms report domain-specific information (model name,
    session ID, tool calls, …) without coupling the storage or TUI layers to
    any particular domain.
    """

    key: str
    """Machine-readable identifier, e.g. ``"model"``, ``"tool_call.0"``."""

    value: str
    """Always a string.  Complex values should be JSON-serialised."""

    category: str = ""
    """Optional grouping hint, e.g. ``"metadata"``, ``"content"``, ``"usage"``."""

    display: str = "kv"
    """Rendering hint for the TUI: ``"kv"``, ``"text"``, ``"json"``, ``"badge"``."""

    source: str = ""
    """Origin phase: ``"request"`` or ``"response"``.  Set automatically by the
    proxy when using ``annotate_request`` / ``annotate_response``."""


class Transform(Protocol):
    name: str
    stores_snapshot: bool

    def transform_request(self, body: dict) -> dict: ...

    def transform_response(self, body: dict, model: str = "") -> dict: ...

    def transform_stream_line(self, line: str, state: dict) -> list[str]: ...

    def rewrite_path(self, path: str) -> str: ...
