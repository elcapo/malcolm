from __future__ import annotations

import json
import logging
import time
import uuid
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from malcolm.storage import RequestRecord
from malcolm.translate import (
    anthropic_request_to_openai,
    anthropic_response_to_openai,
    anthropic_stream_to_openai_lines,
    openai_request_to_anthropic,
    openai_response_to_anthropic,
    openai_stream_to_anthropic_events,
    rewrite_path,
)

if TYPE_CHECKING:
    import httpx

    from malcolm.config import Settings
    from malcolm.storage import NullStorage, Storage

logger = logging.getLogger("malcolm.proxy")

_SKIP_HEADERS = {
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "accept-encoding",
}


def _build_target_url(base_url: str, path: str) -> str:
    """Build the target URL by combining base_url and the request path.

    If base_url contains a path prefix (e.g. /v1) and the request path
    already starts with that same prefix, it is not duplicated.

    Examples:
        ("http://host/v1", "/v1/messages")      -> "http://host/v1/messages"
        ("http://host/v1", "/chat/completions")  -> "http://host/v1/chat/completions"
        ("http://host",    "/v1/chat/completions") -> "http://host/v1/chat/completions"
    """
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    base_path = parsed.path.rstrip("/")

    if base_path and path.startswith(base_path):
        # Client already includes the base path — don't duplicate
        final_path = path
    else:
        # Prepend the base path
        final_path = base_path + path

    origin = f"{parsed.scheme}://{parsed.netloc}"
    return origin + final_path


def _build_headers(
    request: Request, settings: Settings
) -> dict[str, str]:
    """Forward all client headers except hop-by-hop ones.

    If MALCOLM_TARGET_API_KEY is set, override the Authorization header.
    Otherwise forward the client's auth headers as-is (supports both
    OpenAI-style Authorization and Anthropic-style x-api-key).
    """
    headers: dict[str, str] = {}

    for key, value in request.headers.items():
        if key.lower() not in _SKIP_HEADERS:
            headers[key] = value

    if settings.target_api_key:
        headers["authorization"] = f"Bearer {settings.target_api_key}"

    return headers


async def forward_request(
    body: dict,
    request: Request,
    client: httpx.AsyncClient,
    settings: Settings,
    storage: Storage | NullStorage,
) -> JSONResponse:
    record = RequestRecord(
        id=str(uuid.uuid4()),
        model=body.get("model", ""),
        stream=False,
        request_body=body,
    )

    translate = settings.translate

    # Translate request if needed
    if translate == "anthropic_to_openai":
        forwarded_body = anthropic_request_to_openai(body)
    elif translate == "openai_to_anthropic":
        forwarded_body = openai_request_to_anthropic(body)
    else:
        forwarded_body = body

    path = rewrite_path(request.url.path, translate)
    target_url = _build_target_url(settings.target_url, path)
    headers = _build_headers(request, settings)
    start = time.monotonic()

    method = request.method
    try:
        if method in ("POST", "PUT", "PATCH"):
            response = await client.request(method, target_url, json=forwarded_body, headers=headers)
        else:
            response = await client.request(method, target_url, headers=headers)
        record.status_code = response.status_code
        record.duration_ms = (time.monotonic() - start) * 1000

        try:
            response_data = response.json()
        except Exception:
            response_data = {"raw": response.text}

        # Translate response if needed
        if translate == "anthropic_to_openai" and record.status_code == 200:
            record.response_body = openai_response_to_anthropic(response_data, body.get("model", ""))
        elif translate == "openai_to_anthropic" and record.status_code == 200:
            record.response_body = anthropic_response_to_openai(response_data)
        else:
            record.response_body = response_data

        logger.info(
            "request=%s model=%s status=%s duration=%.0fms",
            record.id,
            record.model,
            record.status_code,
            record.duration_ms,
        )
    except Exception as exc:
        record.duration_ms = (time.monotonic() - start) * 1000
        record.error = str(exc)
        logger.error("request=%s error=%s", record.id, exc)
        await storage.save(record)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"Backend error: {exc}", "type": "proxy_error"}},
        )

    await storage.save(record)
    return JSONResponse(status_code=response.status_code, content=record.response_body)


async def forward_request_stream(
    body: dict,
    request: Request,
    client: httpx.AsyncClient,
    settings: Settings,
    storage: Storage | NullStorage,
) -> StreamingResponse:
    record = RequestRecord(
        id=str(uuid.uuid4()),
        model=body.get("model", ""),
        stream=True,
        request_body=body,
    )

    translate = settings.translate

    # Translate request if needed
    if translate == "anthropic_to_openai":
        forwarded_body = anthropic_request_to_openai(body)
    elif translate == "openai_to_anthropic":
        forwarded_body = openai_request_to_anthropic(body)
    else:
        forwarded_body = body

    path = rewrite_path(request.url.path, translate)
    target_url = _build_target_url(settings.target_url, path)
    headers = _build_headers(request, settings)
    start = time.monotonic()

    async def _stream_generator():
        chunks: list[dict] = []
        translate_state: dict = {}
        try:
            async with client.stream(
                "POST", target_url, json=forwarded_body, headers=headers
            ) as response:
                record.status_code = response.status_code

                async for line in response.aiter_lines():
                    if translate == "anthropic_to_openai":
                        # Backend returns OpenAI SSE → translate to Anthropic SSE
                        translated_events = openai_stream_to_anthropic_events(line, translate_state)
                        for event_line in translated_events:
                            yield event_line + "\n"
                    elif translate == "openai_to_anthropic":
                        # Backend returns Anthropic SSE → translate to OpenAI SSE
                        translated_lines = anthropic_stream_to_openai_lines(line, translate_state)
                        for tl in translated_lines:
                            yield tl + "\n"
                    else:
                        yield line + "\n"

                    # Accumulate raw backend chunks for storage
                    if line.startswith("data: ") and line.strip() != "data: [DONE]":
                        try:
                            chunk = json.loads(line[6:])
                            chunks.append(chunk)
                        except json.JSONDecodeError:
                            pass

                yield "\n"

        except Exception as exc:
            record.error = str(exc)
            logger.error("request=%s stream error=%s", record.id, exc)
            error_data = {"error": {"message": f"Backend error: {exc}", "type": "proxy_error"}}
            yield f"data: {json.dumps(error_data)}\n\n"
        finally:
            record.duration_ms = (time.monotonic() - start) * 1000
            if chunks:
                record.response_chunks = chunks
                record.response_body = _assemble_chunks(chunks)
            logger.info(
                "request=%s model=%s stream=true chunks=%d duration=%.0fms",
                record.id,
                record.model,
                len(chunks),
                record.duration_ms,
            )
            await storage.save(record)

    return StreamingResponse(
        _stream_generator(),
        media_type="text/event-stream",
        headers={
            "cache-control": "no-cache",
            "connection": "keep-alive",
            "x-malcolm-request-id": record.id,
        },
    )


def _assemble_chunks(chunks: list[dict]) -> dict:
    """Assemble streaming chunks into a single chat.completion-like response."""
    if not chunks:
        return {}

    first = chunks[0]
    last = chunks[-1]

    assembled_content = ""
    tool_calls_by_index: dict[int, dict] = {}
    finish_reason = None

    for chunk in chunks:
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})

            content = delta.get("content")
            if content:
                assembled_content += content

            if delta.get("tool_calls"):
                for tc in delta["tool_calls"]:
                    idx = tc.get("index", 0)
                    if idx not in tool_calls_by_index:
                        tool_calls_by_index[idx] = {
                            "id": tc.get("id", ""),
                            "type": tc.get("type", "function"),
                            "function": {"name": "", "arguments": ""},
                        }
                    if "function" in tc:
                        if tc["function"].get("name"):
                            tool_calls_by_index[idx]["function"]["name"] = tc["function"]["name"]
                        if tc["function"].get("arguments"):
                            tool_calls_by_index[idx]["function"]["arguments"] += tc["function"]["arguments"]
                    if tc.get("id"):
                        tool_calls_by_index[idx]["id"] = tc["id"]

            fr = choice.get("finish_reason")
            if fr:
                finish_reason = fr

    message: dict = {"role": "assistant", "content": assembled_content or None}
    if tool_calls_by_index:
        message["tool_calls"] = [tool_calls_by_index[i] for i in sorted(tool_calls_by_index)]

    result: dict = {
        "id": first.get("id", ""),
        "object": "chat.completion",
        "created": first.get("created", 0),
        "model": first.get("model", ""),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
    }

    usage = last.get("usage")
    if usage:
        result["usage"] = usage

    return result
