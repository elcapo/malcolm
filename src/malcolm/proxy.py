from __future__ import annotations

import json
import logging
import time
import uuid
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from malcolm.storage import RequestRecord

if TYPE_CHECKING:
    import httpx

    from malcolm.config import Settings
    from malcolm.storage import NullStorage, Storage

logger = logging.getLogger("malcolm.proxy")

_FORWARDED_HEADERS = {
    "x-request-id",
    "http-referer",
    "x-title",
    "user-agent",
}


def _build_target_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    # If the path includes /v1, strip it since base_url typically includes /v1
    if path.startswith("/v1/"):
        path = path[3:]
    elif path.startswith("/v1"):
        path = path[3:] or "/"
    return base + path


def _build_headers(
    request: Request, settings: Settings
) -> dict[str, str]:
    headers: dict[str, str] = {"content-type": "application/json"}

    if settings.target_api_key:
        headers["authorization"] = f"Bearer {settings.target_api_key}"
    else:
        auth = request.headers.get("authorization")
        if auth:
            headers["authorization"] = auth

    for header_name in _FORWARDED_HEADERS:
        value = request.headers.get(header_name)
        if value:
            headers[header_name] = value

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

    target_url = _build_target_url(settings.target_url, request.url.path)
    headers = _build_headers(request, settings)
    start = time.monotonic()

    try:
        response = await client.post(target_url, json=body, headers=headers)
        record.status_code = response.status_code
        record.duration_ms = (time.monotonic() - start) * 1000

        try:
            record.response_body = response.json()
        except Exception:
            record.response_body = {"raw": response.text}

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

    target_url = _build_target_url(settings.target_url, request.url.path)
    headers = _build_headers(request, settings)
    start = time.monotonic()

    async def _stream_generator():
        chunks: list[dict] = []
        try:
            async with client.stream(
                "POST", target_url, json=body, headers=headers
            ) as response:
                record.status_code = response.status_code

                async for line in response.aiter_lines():
                    yield line + "\n"

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
