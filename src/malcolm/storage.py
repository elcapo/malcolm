from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import aiosqlite

def _extract_session_id(request_body: str) -> str:
    """Extract session_id from request metadata, or return 'unknown'."""
    try:
        body = json.loads(request_body) if isinstance(request_body, str) else request_body
        user_id = body.get("metadata", {}).get("user_id", "")
        if isinstance(user_id, str) and user_id.startswith("{"):
            return json.loads(user_id).get("session_id", "unknown")
    except (json.JSONDecodeError, AttributeError):
        pass
    return "unknown"


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS requests (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    model TEXT,
    stream INTEGER NOT NULL DEFAULT 0,
    request_body TEXT NOT NULL,
    response_body TEXT,
    response_chunks TEXT,
    status_code INTEGER,
    duration_ms REAL,
    error TEXT
)
"""


@dataclass
class RequestRecord:
    id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model: str = ""
    stream: bool = False
    request_body: dict = field(default_factory=dict)
    response_body: dict | None = None
    response_chunks: list[dict] | None = None
    status_code: int | None = None
    duration_ms: float | None = None
    error: str | None = None


class Storage:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute(_CREATE_TABLE)
        await self._db.commit()

    async def refresh(self) -> None:
        """End any implicit transaction to see latest writes from other connections."""
        assert self._db is not None
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    async def save(self, record: RequestRecord) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT OR REPLACE INTO requests
                (id, timestamp, model, stream, request_body, response_body,
                 response_chunks, status_code, duration_ms, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.timestamp,
                record.model,
                int(record.stream),
                json.dumps(record.request_body),
                json.dumps(record.response_body) if record.response_body is not None else None,
                json.dumps(record.response_chunks) if record.response_chunks is not None else None,
                record.status_code,
                record.duration_ms,
                record.error,
            ),
        )
        await self._db.commit()

    async def list_sessions(self, limit: int = 50) -> list[dict]:
        """List sessions grouped by session_id from request metadata."""
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        cursor = await self._db.execute(
            "SELECT id, timestamp, model, request_body FROM requests "
            "ORDER BY timestamp DESC LIMIT 1000",
        )
        rows = await cursor.fetchall()

        sessions: dict[str, dict] = {}
        for row in rows:
            row_dict = dict(row)
            session_id = _extract_session_id(row_dict["request_body"])
            if session_id not in sessions:
                body = json.loads(row_dict["request_body"]) if isinstance(row_dict["request_body"], str) else row_dict["request_body"]
                messages = body.get("messages", [])
                user_count = sum(1 for m in messages if m.get("role") == "user")
                sessions[session_id] = {
                    "session_id": session_id,
                    "last_request_id": row_dict["id"],
                    "timestamp": row_dict["timestamp"],
                    "model": row_dict["model"],
                    "user_messages": user_count,
                }

        result = sorted(sessions.values(), key=lambda s: s["timestamp"], reverse=True)
        return result[:limit]

    async def list_recent(self, limit: int = 50) -> list[dict]:
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        cursor = await self._db.execute(
            "SELECT id, timestamp, model, stream, status_code, duration_ms, error "
            "FROM requests ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get(self, record_id: str) -> dict | None:
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        cursor = await self._db.execute(
            "SELECT * FROM requests WHERE id = ?", (record_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        result = dict(row)
        for key in ("request_body", "response_body", "response_chunks"):
            if result.get(key) is not None:
                result[key] = json.loads(result[key])
        return result

    async def delete(self, record_id: str) -> bool:
        assert self._db is not None
        cursor = await self._db.execute(
            "DELETE FROM requests WHERE id = ?", (record_id,)
        )
        await self._db.commit()
        return cursor.rowcount > 0


class NullStorage:
    """No-op storage used when persistence is disabled."""

    async def init(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def save(self, record: RequestRecord) -> None:
        pass

    async def list_recent(self, limit: int = 50) -> list[dict]:
        return []

    async def get(self, record_id: str) -> dict | None:
        return None

    async def delete(self, record_id: str) -> bool:
        return False
