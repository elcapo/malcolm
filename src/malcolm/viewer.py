from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

router = APIRouter()


@router.get("/")
async def index():
    return RedirectResponse(url="/logs")


@router.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request):
    storage = request.app.state.storage
    records = await storage.list_recent(limit=100)

    rows = ""
    for r in records:
        status_class = "ok" if r.get("status_code") == 200 else "err"
        duration = f'{r.get("duration_ms", 0):.0f}ms' if r.get("duration_ms") else "-"
        stream_badge = "SSE" if r.get("stream") else ""
        error_icon = " !" if r.get("error") else ""
        rows += f"""
        <tr class="{status_class}">
            <td><a href="/logs/{r['id']}">{r['id'][:8]}...</a></td>
            <td>{r.get('timestamp', '')[:19]}</td>
            <td>{r.get('model', '')}</td>
            <td>{stream_badge}</td>
            <td>{r.get('status_code', '-')}</td>
            <td>{duration}</td>
            <td>{error_icon}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>malcolm - logs</title>
    <style>
        body {{ background: #1a1a2e; color: #e0e0e0; font-family: monospace; margin: 2rem; }}
        h1 {{ color: #e94560; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ padding: 0.5rem 1rem; text-align: left; border-bottom: 1px solid #333; }}
        th {{ color: #e94560; }}
        a {{ color: #0f9b8e; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .err td {{ color: #e94560; }}
        .empty {{ color: #666; padding: 2rem; text-align: center; }}
    </style>
</head>
<body>
    <h1>malcolm</h1>
    {"<table><tr><th>ID</th><th>Time</th><th>Model</th><th>Stream</th><th>Status</th><th>Duration</th><th></th></tr>"
     + rows + "</table>" if rows else '<p class="empty">No requests logged yet.</p>'}
</body>
</html>"""
    return HTMLResponse(content=html)


@router.get("/logs/{record_id}", response_class=HTMLResponse)
async def log_detail(request: Request, record_id: str):
    storage = request.app.state.storage
    record = await storage.get(record_id)

    if record is None:
        return HTMLResponse(content="<h1>Not found</h1>", status_code=404)

    req_json = json.dumps(record.get("request_body"), indent=2, ensure_ascii=False)
    resp_json = json.dumps(record.get("response_body"), indent=2, ensure_ascii=False) if record.get("response_body") else "null"
    chunks_json = json.dumps(record.get("response_chunks"), indent=2, ensure_ascii=False) if record.get("response_chunks") else None

    error_section = f'<h2>Error</h2><pre class="error">{record.get("error")}</pre>' if record.get("error") else ""
    chunks_section = f"<h2>Streaming Chunks ({len(record.get('response_chunks', []))})</h2><pre>{chunks_json}</pre>" if chunks_json else ""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>malcolm - {record_id[:8]}</title>
    <style>
        body {{ background: #1a1a2e; color: #e0e0e0; font-family: monospace; margin: 2rem; }}
        h1 {{ color: #e94560; }}
        h2 {{ color: #0f9b8e; margin-top: 2rem; }}
        pre {{ background: #16213e; padding: 1rem; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; }}
        .meta {{ color: #888; margin-bottom: 1rem; }}
        .meta span {{ margin-right: 2rem; }}
        a {{ color: #0f9b8e; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .error {{ color: #e94560; }}
    </style>
</head>
<body>
    <a href="/logs">&lt; back</a>
    <h1>{record_id}</h1>
    <div class="meta">
        <span>Model: {record.get('model', '-')}</span>
        <span>Status: {record.get('status_code', '-')}</span>
        <span>Duration: {record.get('duration_ms', 0):.0f}ms</span>
        <span>Stream: {'yes' if record.get('stream') else 'no'}</span>
        <span>Time: {record.get('timestamp', '')}</span>
    </div>
    {error_section}
    <h2>Request</h2>
    <pre>{req_json}</pre>
    <h2>Response</h2>
    <pre>{resp_json}</pre>
    {chunks_section}
</body>
</html>"""
    return HTMLResponse(content=html)


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
