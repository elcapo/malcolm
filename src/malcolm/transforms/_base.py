"""Transform protocol — the interface every transform must implement."""

from __future__ import annotations

from typing import Protocol


class Transform(Protocol):
    name: str

    def transform_request(self, body: dict) -> dict: ...

    def transform_response(self, body: dict, model: str = "") -> dict: ...

    def transform_stream_line(self, line: str, state: dict) -> list[str]: ...

    def rewrite_path(self, path: str) -> str: ...
