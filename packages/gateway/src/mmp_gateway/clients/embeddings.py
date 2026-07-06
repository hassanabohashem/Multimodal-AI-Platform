"""Async client for the SigLIP-2 embedding service."""
from __future__ import annotations

import httpx

from mmp_gateway.errors import UpstreamUnavailable


class EmbeddingsClient:
    """Embeds text or images by calling the internal embedding service."""

    def __init__(self, base_url: str, timeout_s: float, internal_token: str) -> None:
        self._http = httpx.AsyncClient(
            base_url=base_url, timeout=timeout_s,
            headers={"X-Internal-Token": internal_token},
        )

    async def aclose(self) -> None:
        """Close the connection pool."""
        await self._http.aclose()

    async def embed_text(self, text: str) -> list[float]:
        """Return one L2-normalized text embedding."""
        try:
            r = await self._http.post("/embed/text", json={"texts": [text]})
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise UpstreamUnavailable(f"embedding service error: {exc}") from exc
        return list(r.json()["embeddings"][0])

    async def embed_image(self, jpeg: bytes) -> list[float]:
        """Return one L2-normalized image embedding."""
        try:
            r = await self._http.post("/embed/image", files={"file": ("q.jpg", jpeg, "image/jpeg")})
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise UpstreamUnavailable(f"embedding service error: {exc}") from exc
        return list(r.json()["embeddings"][0])
