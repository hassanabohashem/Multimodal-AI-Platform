"""Structured JSON logging shared by every service."""
from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(service: str, level: str = "INFO") -> structlog.stdlib.BoundLogger:
    """Configure structlog to emit JSON lines with a bound service name."""
    logging.basicConfig(stream=sys.stdout, level=level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger().bind(service=service)
