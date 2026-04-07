"""
transforms.py — pluggable request/response transform pipeline for Malcolm.

Each Transform modifies the request before it reaches the backend and/or
modifies the response before it reaches the client.  The proxy applies
transforms in order on the request side and in reverse order on the
response side.

The pipeline is assembled at startup via ``build_pipeline(settings)``.
"""

from __future__ import annotations

import json
from typing import Protocol

from malcolm.ghostkey import obfuscate, restore, scan_request
from malcolm.translate import (
    anthropic_request_to_openai,
    anthropic_response_to_openai,
    anthropic_stream_to_openai_lines,
    openai_request_to_anthropic,
    openai_response_to_anthropic,
    openai_stream_to_anthropic_events,
    rewrite_path,
)

from malcolm.config import Settings


class Transform(Protocol):
    name: str

    def transform_request(self, body: dict) -> dict: ...

    def transform_response(self, body: dict, model: str = "") -> dict: ...

    def transform_stream_line(self, line: str, state: dict) -> list[str]: ...

    def rewrite_path(self, path: str) -> str: ...


class GhostKeyTransform:
    name = "ghostkey"

    def transform_request(self, body: dict) -> dict:
        text = json.dumps(body)
        scan_request(text)
        clean = obfuscate(text)
        if clean == text:
            return body
        return json.loads(clean)

    def transform_response(self, body: dict, model: str = "") -> dict:
        text = json.dumps(body)
        restored = restore(text)
        if restored == text:
            return body
        return json.loads(restored)

    def transform_stream_line(self, line: str, state: dict) -> list[str]:
        return [restore(line)]

    def rewrite_path(self, path: str) -> str:
        return path


class TranslationTransform:
    def __init__(self, direction: str) -> None:
        self._direction = direction
        self.name = "translation"

    def transform_request(self, body: dict) -> dict:
        if self._direction == "anthropic_to_openai":
            return anthropic_request_to_openai(body)
        if self._direction == "openai_to_anthropic":
            return openai_request_to_anthropic(body)
        return body

    def transform_response(self, body: dict, model: str = "") -> dict:
        if self._direction == "anthropic_to_openai":
            return openai_response_to_anthropic(body, model)
        if self._direction == "openai_to_anthropic":
            return anthropic_response_to_openai(body)
        return body

    def transform_stream_line(self, line: str, state: dict) -> list[str]:
        if self._direction == "anthropic_to_openai":
            return openai_stream_to_anthropic_events(line, state)
        if self._direction == "openai_to_anthropic":
            return anthropic_stream_to_openai_lines(line, state)
        return [line]

    def rewrite_path(self, path: str) -> str:
        return rewrite_path(path, self._direction)


def build_pipeline(settings: Settings) -> list[Transform]:
    pipeline: list[Transform] = []
    if settings.ghostkey_enabled:
        pipeline.append(GhostKeyTransform())
    if settings.translate:
        pipeline.append(TranslationTransform(settings.translate))
    return pipeline
