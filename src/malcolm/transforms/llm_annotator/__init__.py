"""LLM annotator — extracts LLM-specific metadata as generic annotations.

This transform is a pass-through: it never modifies request or response
bodies.  Instead it implements ``annotate_request()`` and
``annotate_response()`` to inspect the original data and produce
:class:`Annotation` objects that the TUI (or any other consumer) can
display without knowing LLM specifics.
"""

from __future__ import annotations

from malcolm.formats import extract_session_hint, parse_record
from malcolm.transforms._base import Annotation


class LLMAnnotatorTransform:
    name = "llm_annotator"
    stores_snapshot = False

    # ── Pass-through transform methods ───────────────────────────────

    def transform_request(self, body: dict) -> dict:
        return body

    def transform_response(self, body: dict, model: str = "") -> dict:
        return body

    def transform_stream_line(self, line: str, state: dict) -> list[str]:
        return [line]

    def rewrite_path(self, path: str) -> str:
        return path

    # ── Request annotations ──────────────────────────────────────────

    def annotate_request(
        self,
        request_body: dict,
        request_headers: dict | None = None,
    ) -> list[Annotation]:
        annotations: list[Annotation] = []

        # Model
        model = request_body.get("model", "")
        if model:
            annotations.append(Annotation("model", model, "metadata", "badge"))

        # Stream
        if request_body.get("stream"):
            annotations.append(Annotation("stream", "true", "metadata", "badge"))

        # Session ID
        hint = extract_session_hint(request_body, request_headers)
        if hint:
            annotations.append(Annotation("session_id", hint, "metadata", "kv"))

        # Parse request messages
        record = {"request_body": request_body, "response_body": {}, "response_chunks": []}
        conversation = parse_record(record)

        user_idx = 0
        for msg in conversation.messages:
            if msg.role == "system":
                annotations.append(
                    Annotation("system_prompt", msg.text or "", "content", "text"),
                )
            elif msg.role == "user":
                annotations.append(
                    Annotation(f"user_message.{user_idx}", msg.text or "", "content", "text"),
                )
                user_idx += 1

        return annotations

    # ── Response annotations ─────────────────────────────────────────

    def annotate_response(
        self,
        response_body: dict | None,
        response_chunks: list[dict] | None = None,
    ) -> list[Annotation]:
        annotations: list[Annotation] = []

        if not response_body and not response_chunks:
            return annotations

        # Parse response messages
        record = {
            "request_body": {},
            "response_body": response_body,
            "response_chunks": response_chunks or [],
        }
        conversation = parse_record(record)

        assistant_idx = 0
        tool_idx = 0
        for msg in conversation.messages:
            if msg.role == "assistant":
                if msg.text:
                    annotations.append(
                        Annotation(
                            f"assistant_message.{assistant_idx}",
                            msg.text, "content", "text",
                        ),
                    )
                    assistant_idx += 1
                for tc in msg.tool_calls:
                    annotations.append(
                        Annotation(f"tool_call.{tool_idx}", tc.name, "content", "kv"),
                    )
                    tool_idx += 1
            elif msg.role == "tool":
                if msg.tool_result:
                    annotations.append(
                        Annotation(
                            f"tool_result.{tool_idx}",
                            msg.tool_result[:500], "content", "text",
                        ),
                    )

        # Usage
        if response_body:
            usage = response_body.get("usage", {})
            if usage:
                for src_key, ann_key in [
                    ("prompt_tokens", "input_tokens"),
                    ("completion_tokens", "output_tokens"),
                    ("input_tokens", "input_tokens"),
                    ("output_tokens", "output_tokens"),
                ]:
                    val = usage.get(src_key)
                    if val is not None:
                        annotations.append(
                            Annotation(ann_key, str(val), "usage", "kv"),
                        )

        return annotations


def create(config: dict) -> LLMAnnotatorTransform:
    return LLMAnnotatorTransform()
