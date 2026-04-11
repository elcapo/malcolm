from __future__ import annotations

import json
import logging
import time
import uuid
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from malcolm.formats import assemble_chunks
from malcolm.storage import RequestRecord, TransformRecord

if TYPE_CHECKING:
    import httpx

    from malcolm.config import Settings
    from malcolm.storage import NullStorage, Storage
    from malcolm.transforms import Annotator, Transform

logger = logging.getLogger("malcolm.proxy")

_SKIP_HEADERS = {
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "accept-encoding",
}

# Headers captured from client requests for session detection and diagnostics.
# Auth headers are intentionally excluded.
_CAPTURE_HEADERS = {"user-agent", "x-session-affinity", "anthropic-beta", "x-app"}


def _build_target_url(base_url: str, path: str, query: str = "") -> str:
    """Build the target URL by combining base_url, the request path and query.

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
    url = origin + final_path
    if query:
        url = f"{url}?{query}"
    return url


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


async def _run_request_annotators(
    annotators: list[Annotator],
    record_id: str,
    request_body: dict,
    request_headers: dict,
    storage: Storage | NullStorage,
) -> None:
    """Invoke ``annotate_request()`` on every annotator in the pipeline."""
    for ann in annotators:
        try:
            annotations = ann.annotate_request(request_body, request_headers)
            for a in annotations:
                a.source = "request"
            if annotations:
                await storage.save_annotations(record_id, ann.name, annotations)
        except Exception as exc:
            logger.warning(
                "request=%s annotate_request %s failed: %s",
                record_id, ann.name, exc,
            )


async def _run_response_annotators(
    annotators: list[Annotator],
    record_id: str,
    response_body: dict | None,
    response_chunks: list[dict] | None,
    storage: Storage | NullStorage,
) -> None:
    """Invoke ``annotate_response()`` on every annotator in the pipeline."""
    for ann in annotators:
        try:
            annotations = ann.annotate_response(response_body, response_chunks)
            for a in annotations:
                a.source = "response"
            if annotations:
                await storage.save_annotations(record_id, ann.name, annotations)
        except Exception as exc:
            logger.warning(
                "request=%s annotate_response %s failed: %s",
                record_id, ann.name, exc,
            )


async def forward_request(
    body: dict,
    request: Request,
    client: httpx.AsyncClient,
    settings: Settings,
    storage: Storage | NullStorage,
    transforms: list[Transform] | None = None,
    annotators: list[Annotator] | None = None,
) -> JSONResponse:
    captured_headers = {
        k: v for k, v in request.headers.items() if k.lower() in _CAPTURE_HEADERS
    }
    record = RequestRecord(
        id=str(uuid.uuid4()),
        model=body.get("model", ""),
        stream=False,
        request_body=body,
        request_headers=captured_headers,
    )

    if transforms is None:
        transforms = []
    if annotators is None:
        annotators = []

    # ── Apply request transforms (forward order) ──────────────────
    transform_snapshots: dict[str, dict] = {}
    forwarded_body = body
    for t in transforms:
        forwarded_body = t.transform_request(forwarded_body)
        if t.stores_snapshot:
            transform_snapshots[t.name] = {"request_body": forwarded_body}

    # ── Resolve path & forward ────────────────────────────────────
    path = request.url.path
    for t in transforms:
        path = t.rewrite_path(path)
    target_url = _build_target_url(settings.target_url, path, request.url.query)
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

        # Store raw backend response
        record.response_body = response_data

        # ── Apply response transforms (reverse order) ─────────────
        # Transforms run on every status code so passes like ghostkey can
        # restore originals inside error messages.  Transforms that only
        # make sense on success bodies (e.g. translation) detect error
        # shapes and pass through unchanged.
        client_response = response_data
        model = body.get("model", "")
        for t in reversed(transforms):
            client_response = t.transform_response(client_response, model=model)
            if t.name in transform_snapshots:
                transform_snapshots[t.name]["response_body"] = client_response

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
        for name, snapshot in transform_snapshots.items():
            await storage.save_transform(TransformRecord(
                request_id=record.id, transform_type=name, **snapshot,
            ))
        await _run_request_annotators(
            annotators, record.id, body, captured_headers, storage,
        )
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"Backend error: {exc}", "type": "proxy_error"}},
        )

    await storage.save(record)
    for name, snapshot in transform_snapshots.items():
        await storage.save_transform(TransformRecord(
            request_id=record.id, transform_type=name, **snapshot,
        ))
    await _run_request_annotators(
        annotators, record.id, body, captured_headers, storage,
    )
    await _run_response_annotators(
        annotators, record.id, record.response_body, record.response_chunks,
        storage,
    )
    return JSONResponse(status_code=response.status_code, content=client_response)


async def forward_request_stream(
    body: dict,
    request: Request,
    client: httpx.AsyncClient,
    settings: Settings,
    storage: Storage | NullStorage,
    transforms: list[Transform] | None = None,
    annotators: list[Annotator] | None = None,
) -> StreamingResponse:
    captured_headers = {
        k: v for k, v in request.headers.items() if k.lower() in _CAPTURE_HEADERS
    }
    record = RequestRecord(
        id=str(uuid.uuid4()),
        model=body.get("model", ""),
        stream=True,
        request_body=body,
        request_headers=captured_headers,
    )

    if transforms is None:
        transforms = []
    if annotators is None:
        annotators = []

    # ── Apply request transforms (forward order) ──────────────────
    transform_snapshots: dict[str, dict] = {}
    forwarded_body = body
    for t in transforms:
        forwarded_body = t.transform_request(forwarded_body)
        if t.stores_snapshot:
            transform_snapshots[t.name] = {"request_body": forwarded_body}

    # ── Resolve path & forward ────────────────────────────────────
    path = request.url.path
    for t in transforms:
        path = t.rewrite_path(path)
    target_url = _build_target_url(settings.target_url, path, request.url.query)
    headers = _build_headers(request, settings)
    start = time.monotonic()

    async def _stream_generator():
        raw_chunks: list[dict] = []
        stream_states: dict[str, dict] = {t.name: {} for t in transforms}
        try:
            async with client.stream(
                "POST", target_url, json=forwarded_body, headers=headers
            ) as response:
                record.status_code = response.status_code

                async for line in response.aiter_lines():
                    # Accumulate raw backend chunks for storage
                    if line.startswith("data: ") and line.strip() != "data: [DONE]":
                        try:
                            raw_chunks.append(json.loads(line[6:]))
                        except json.JSONDecodeError:
                            pass

                    # Apply stream transforms in reverse order
                    output_lines = [line]
                    for t in reversed(transforms):
                        next_lines: list[str] = []
                        for ol in output_lines:
                            next_lines.extend(t.transform_stream_line(ol, stream_states[t.name]))
                        output_lines = next_lines

                    for ol in output_lines:
                        yield ol + "\n"

                yield "\n"

        except Exception as exc:
            record.error = str(exc)
            logger.error("request=%s stream error=%s", record.id, exc)
            error_data = {"error": {"message": f"Backend error: {exc}", "type": "proxy_error"}}
            yield f"data: {json.dumps(error_data)}\n\n"
        finally:
            record.duration_ms = (time.monotonic() - start) * 1000
            if raw_chunks:
                record.response_chunks = raw_chunks
                record.response_body = assemble_chunks(raw_chunks)
            logger.info(
                "request=%s model=%s stream=true chunks=%d duration=%.0fms",
                record.id,
                record.model,
                len(raw_chunks),
                record.duration_ms,
            )
            await storage.save(record)
            for name, snapshot in transform_snapshots.items():
                await storage.save_transform(TransformRecord(
                    request_id=record.id, transform_type=name, **snapshot,
                ))
            await _run_request_annotators(
                annotators, record.id, body, captured_headers, storage,
            )
            await _run_response_annotators(
                annotators, record.id, record.response_body,
                record.response_chunks, storage,
            )

    return StreamingResponse(
        _stream_generator(),
        media_type="text/event-stream",
        headers={
            "cache-control": "no-cache",
            "connection": "keep-alive",
            "x-malcolm-request-id": record.id,
        },
    )


