"""Request-ID propagation, structured log context, and Prometheus timing."""
from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import Request, Response
from prometheus_client import Counter, Histogram

REQUESTS = Counter("http_requests_total", "Requests", ["route", "method", "status"])
LATENCY = Histogram(
    "http_request_duration_seconds", "Latency", ["route"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32),
)


async def observability_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Bind a request ID, time the request, and record metrics."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        structlog.contextvars.clear_contextvars()
    elapsed = time.perf_counter() - start
    route = request.scope.get("route").path if request.scope.get("route") else request.url.path
    REQUESTS.labels(route, request.method, response.status_code).inc()
    LATENCY.labels(route).observe(elapsed)
    response.headers["X-Request-ID"] = request_id
    return response
