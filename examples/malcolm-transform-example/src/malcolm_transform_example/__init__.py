"""
malcolm_transform_example — pass-through transform that logs request/response structure.

Demonstrates how to publish an external Malcolm transform as a standalone pip
package. Registered via the ``malcolm.transforms`` entry point in pyproject.toml.
"""

from __future__ import annotations

import json
import logging

__all__ = ["HeaderLoggerTransform", "create"]


class HeaderLoggerTransform:
    name = "header_logger"

    def __init__(
        self,
        prefix: str = "[header_logger]",
        logger_name: str = "malcolm_transform_example",
    ) -> None:
        self._prefix = prefix
        self._log = logging.getLogger(logger_name)

    def transform_request(self, body: dict) -> dict:
        keys = sorted(body.keys()) if isinstance(body, dict) else []
        size = len(json.dumps(body, default=str))
        self._log.info("%s request keys=%s size=%d", self._prefix, keys, size)
        return body

    def transform_response(self, body: dict, model: str = "") -> dict:
        size = len(json.dumps(body, default=str))
        self._log.info("%s response model=%s size=%d", self._prefix, model, size)
        return body

    def transform_stream_line(self, line: str, state: dict) -> list[str]:
        state["lines_seen"] = state.get("lines_seen", 0) + 1
        return [line]

    def rewrite_path(self, path: str) -> str:
        return path


def create(config: dict) -> HeaderLoggerTransform:
    return HeaderLoggerTransform(
        prefix=config.get("prefix", "[header_logger]"),
        logger_name=config.get("logger_name", "malcolm_transform_example"),
    )
