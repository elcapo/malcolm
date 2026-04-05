from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()

_templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@router.get("/")
async def index():
    return RedirectResponse(url="/logs")


@router.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request):
    storage = request.app.state.storage
    records = await storage.list_recent(limit=100)

    return _templates.TemplateResponse(request, "logs.html", {"rows": records})


@router.get("/logs/{record_id}", response_class=HTMLResponse)
async def log_detail(request: Request, record_id: str):
    storage = request.app.state.storage
    record = await storage.get(record_id)

    if record is None:
        return HTMLResponse(content="<h1>Not found</h1>", status_code=404)

    req_json = json.dumps(record.get("request_body"), indent=2, ensure_ascii=False)
    resp_json = json.dumps(record.get("response_body"), indent=2, ensure_ascii=False) if record.get("response_body") else "null"
    chunks_json = json.dumps(record.get("response_chunks"), indent=2, ensure_ascii=False) if record.get("response_chunks") else None

    return _templates.TemplateResponse(request, "detail.html", {
        "record_id": record_id,
        "model": record.get("model", "-"),
        "status_code": record.get("status_code", "-"),
        "duration_ms": record.get("duration_ms"),
        "stream": record.get("stream"),
        "timestamp": record.get("timestamp", ""),
        "error": record.get("error"),
        "req_json": req_json,
        "resp_json": resp_json,
        "chunks_json": chunks_json,
        "chunks_count": len(record.get("response_chunks", []) or []),
    })


@router.get("/api/logs")
async def api_logs(request: Request, limit: int = 50):
    storage = request.app.state.storage
    records = await storage.list_recent(limit=limit)
    return JSONResponse(content=records)


@router.get("/api/logs/{record_id}")
async def api_log_detail(request: Request, record_id: str):
    storage = request.app.state.storage
    record = await storage.get(record_id)
    if record is None:
        return JSONResponse(content={"error": "not found"}, status_code=404)
    return JSONResponse(content=record)
