"""Translation transform — converts between Anthropic and OpenAI API formats."""

from __future__ import annotations

from malcolm.transforms.translation.engine import (
    anthropic_request_to_openai,
    anthropic_response_to_openai,
    anthropic_stream_to_openai_lines,
    openai_request_to_anthropic,
    openai_response_to_anthropic,
    openai_stream_to_anthropic_events,
    rewrite_path,
)


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


def create(config: dict) -> TranslationTransform:
    direction = config.get("direction", "")
    if not direction:
        raise ValueError(
            "translation transform requires 'direction' "
            "(e.g. 'anthropic_to_openai' or 'openai_to_anthropic')"
        )
    return TranslationTransform(direction)
