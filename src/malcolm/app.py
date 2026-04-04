from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from malcolm.config import Settings
from malcolm.proxy import forward_request, forward_request_stream
from malcolm.storage import NullStorage, Storage
from malcolm.viewer import router as viewer_router

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

        logging.basicConfig(
            level=getattr(logging, settings.log_level.upper(), logging.INFO),
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )

        logger.info(
            "malcolm started — target=%s storage=%s",
            settings.target_url,
            "enabled" if settings.storage_enabled else "disabled",
        )

        yield

        await app.state.client.aclose()
        await app.state.storage.close()

    app = FastAPI(title="malcolm", lifespan=lifespan)
    app.include_router(viewer_router)

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        stream = body.get("stream", False)

        if stream:
            return await forward_request_stream(
                body, request, request.app.state.client,
                request.app.state.settings, request.app.state.storage,
            )
        else:
            return await forward_request(
                body, request, request.app.state.client,
                request.app.state.settings, request.app.state.storage,
            )

    @app.get("/v1/models")
    @app.get("/models")
    async def list_models(request: Request):
        s = request.app.state.settings
        target_url = s.target_url.rstrip("/") + "/models"
        headers = {}
        if s.target_api_key:
            headers["authorization"] = f"Bearer {s.target_api_key}"
        else:
            auth = request.headers.get("authorization")
            if auth:
                headers["authorization"] = auth

        try:
            response = await request.app.state.client.get(target_url, headers=headers)
            return JSONResponse(status_code=response.status_code, content=response.json())
        except Exception as exc:
            return JSONResponse(
                status_code=502,
                content={"error": {"message": f"Backend error: {exc}", "type": "proxy_error"}},
            )

    return app
