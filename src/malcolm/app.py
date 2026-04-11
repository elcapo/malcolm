from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response

from malcolm.config import Settings
from malcolm.proxy import forward_request, forward_request_stream
from malcolm.storage import NullStorage, Storage
from malcolm.transforms import build_pipeline

logger = logging.getLogger("malcolm")


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings

        app.state.client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))

        if settings.storage_enabled:
            storage = Storage(settings.db_path)
            await storage.init()
        else:
            storage = NullStorage()
            await storage.init()
        app.state.storage = storage

        logger.info(
            "malcolm started — target=%s storage=%s",
            settings.target_url,
            "enabled" if settings.storage_enabled else "disabled",
        )

        yield

        await app.state.client.aclose()
        await app.state.storage.close()

    app = FastAPI(title="malcolm", lifespan=lifespan)

    pipeline = build_pipeline(settings.config_file)

    @app.head("/")
    async def health_check():
        return Response(status_code=200)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
    async def catch_all(request: Request, path: str):
        """Catch-all proxy: forwards any unmatched route to the backend."""
        if request.method in ("POST", "PUT", "PATCH"):
            body = await request.json()
            stream = body.get("stream", False)

            if stream:
                return await forward_request_stream(
                    body, request, request.app.state.client,
                    request.app.state.settings, request.app.state.storage,
                    transforms=pipeline.transforms,
                    annotators=pipeline.annotators,
                )
            else:
                return await forward_request(
                    body, request, request.app.state.client,
                    request.app.state.settings, request.app.state.storage,
                    transforms=pipeline.transforms,
                    annotators=pipeline.annotators,
                )
        else:
            # GET, DELETE, HEAD, OPTIONS — forward without body
            return await forward_request(
                {}, request, request.app.state.client,
                request.app.state.settings, request.app.state.storage,
                transforms=pipeline.transforms,
                annotators=pipeline.annotators,
            )

    return app
