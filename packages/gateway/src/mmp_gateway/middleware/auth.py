"""API-key dependency for write/inference endpoints."""
from __future__ import annotations

from fastapi import Header, HTTPException

from mmp_gateway.settings import settings


async def require_api_key(x_api_key: str = Header(default="")) -> None:
    """Reject requests without a valid X-API-Key header."""
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
