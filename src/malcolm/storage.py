from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import aiosqlite


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

_CREATE_TRANSFORMS_TABLE = """
CREATE TABLE IF NOT EXISTS request_transforms (
    request_id TEXT NOT NULL REFERENCES requests(id) ON DELETE CASCADE,
    transform_type TEXT NOT NULL,
    request_body TEXT,
    response_body TEXT,
    response_chunks TEXT,
    PRIMARY KEY (request_id, transform_type)
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


@dataclass
class TransformRecord:
    request_id: str
    transform_type: str
    request_body: dict | None = None
    response_body: dict | None = None
    response_chunks: list[dict] | None = None


class Storage:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys = ON")
        await self._db.execute(_CREATE_TABLE)
        await self._db.execute(_CREATE_TRANSFORMS_TABLE)
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_requests_timestamp "
            "ON requests (timestamp DESC)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_transforms_request_id "
            "ON request_transforms (request_id)"
        )
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

    async def list_page_full(
        self, page_size: int = 50, before: str | None = None,
    ) -> list[dict]:
        """Fetch a page of records with request_body, ordered by timestamp DESC.

        Uses cursor-based pagination: ``before`` is an ISO timestamp; only
        records strictly older than it are returned.
        """
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        if before:
            cursor = await self._db.execute(
                "SELECT id, timestamp, model, stream, status_code, duration_ms, "
                "error, request_body "
                "FROM requests WHERE timestamp < ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (before, page_size),
            )
        else:
            cursor = await self._db.execute(
                "SELECT id, timestamp, model, stream, status_code, duration_ms, "
                "error, request_body "
                "FROM requests ORDER BY timestamp DESC LIMIT ?",
                (page_size,),
            )
        rows = await cursor.fetchall()
        result = []
        for row in rows:
            d = dict(row)
            if d.get("request_body") is not None:
                d["request_body"] = json.loads(d["request_body"])
            result.append(d)
        return result

    async def save_transform(self, transform: TransformRecord) -> None:
        assert self._db is not None
        await self._db.execute(
            """
            INSERT OR REPLACE INTO request_transforms
                (request_id, transform_type, request_body, response_body, response_chunks)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                transform.request_id,
                transform.transform_type,
                json.dumps(transform.request_body) if transform.request_body is not None else None,
                json.dumps(transform.response_body) if transform.response_body is not None else None,
                json.dumps(transform.response_chunks) if transform.response_chunks is not None else None,
            ),
        )
        await self._db.commit()

    async def get_transforms(self, request_id: str) -> list[dict]:
        assert self._db is not None
        self._db.row_factory = aiosqlite.Row
        cursor = await self._db.execute(
            "SELECT transform_type, request_body, response_body, response_chunks "
            "FROM request_transforms WHERE request_id = ?",
            (request_id,),
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            d = dict(row)
            for key in ("request_body", "response_body", "response_chunks"):
                if d.get(key) is not None:
                    d[key] = json.loads(d[key])
            results.append(d)
        return results

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
        result["transforms"] = await self.get_transforms(record_id)
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

    async def save_transform(self, transform: TransformRecord) -> None:
        pass

    async def get_transforms(self, request_id: str) -> list[dict]:
        return []

    async def list_recent(self, limit: int = 50) -> list[dict]:
        return []

    async def list_page_full(
        self, page_size: int = 50, before: str | None = None,
    ) -> list[dict]:
        return []

    async def get(self, record_id: str) -> dict | None:
        return None

    async def delete(self, record_id: str) -> bool:
        return False
