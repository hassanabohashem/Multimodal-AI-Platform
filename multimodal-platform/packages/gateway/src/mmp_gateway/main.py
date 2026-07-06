"""Gateway application factory.

The gateway owns no model weights. It validates, authenticates, routes,
measures, and translates failures into problem+json.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from mmp_common.logging import configure_logging
from mmp_gateway.clients.embeddings import EmbeddingsClient
from mmp_gateway.clients.ocr import OCRClient
from mmp_gateway.clients.search import SearchStore
from mmp_gateway.clients.vllm import VLMClient
from mmp_gateway.errors import install_handlers
from mmp_gateway.middleware.observability import observability_middleware
from mmp_gateway.routers import caption, ocr, search, vqa
from mmp_gateway.settings import settings

log = configure_logging("gateway")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared clients at startup; close them at shutdown."""
    app.state.vlm = VLMClient(settings.vllm_url, settings.timeout_caption_s)
    app.state.embeddings = EmbeddingsClient(
        settings.embeddings_url, settings.timeout_search_s, settings.internal_token
    )
    app.state.ocr = OCRClient(settings.ocr_url, settings.timeout_ocr_s, settings.internal_token)
    app.state.store = SearchStore(settings.qdrant_url)
    await app.state.store.ensure_collection()
    log.info("gateway ready")
    yield
    await app.state.vlm.aclose()
    await app.state.embeddings.aclose()
    await app.state.ocr.aclose()


def create_app() -> FastAPI:
    """Build the FastAPI app with middleware, routers, and error handlers."""
    app = FastAPI(title="Multimodal AI Platform", version="0.1.0", lifespan=lifespan)
    limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)
    app.middleware("http")(observability_middleware)
    install_handlers(app)
    for r in (caption.router, vqa.router, search.router, ocr.router):
        app.include_router(r)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
