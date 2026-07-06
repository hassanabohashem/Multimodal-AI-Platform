"""RFC 9457 problem+json error handling."""
from __future__ import annotations

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from mmp_common.images import ImageValidationError
from mmp_common.schemas import Problem

log = structlog.get_logger()


class UpstreamUnavailable(Exception):
    """A dependent service is down or the circuit breaker is open."""


def problem(request: Request, status: int, title: str, detail: str | None = None) -> JSONResponse:
    """Build a problem+json response carrying the request ID."""
    body = Problem(
        title=title,
        status=status,
        detail=detail,
        request_id=getattr(request.state, "request_id", None),
    )
    return JSONResponse(status_code=status, content=body.model_dump(), media_type="application/problem+json")


def install_handlers(app: FastAPI) -> None:
    """Register exception handlers; clients never see stack traces."""

    @app.exception_handler(ImageValidationError)
    async def _img(request: Request, exc: ImageValidationError) -> JSONResponse:
        return problem(request, 422, "invalid image", str(exc))

    @app.exception_handler(UpstreamUnavailable)
    async def _up(request: Request, exc: UpstreamUnavailable) -> JSONResponse:
        resp = problem(request, 503, "model temporarily unavailable", str(exc))
        resp.headers["Retry-After"] = "10"
        return resp

    @app.exception_handler(Exception)
    async def _any(request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled", error=str(exc), request_id=getattr(request.state, "request_id", None))
        return problem(request, 500, "internal error")
