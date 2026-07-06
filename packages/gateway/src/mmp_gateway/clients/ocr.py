"""Async client for the OCR service."""
from __future__ import annotations

import httpx

from mmp_common.schemas import OCRResponse
from mmp_gateway.errors import UpstreamUnavailable


class OCRClient:
    """Forwards documents to the PaddleOCR-VL service."""

    def __init__(self, base_url: str, timeout_s: float, internal_token: str) -> None:
        self._http = httpx.AsyncClient(
            base_url=base_url, timeout=timeout_s,
            headers={"X-Internal-Token": internal_token},
        )

    async def aclose(self) -> None:
        """Close the connection pool."""
        await self._http.aclose()

    async def extract(self, data: bytes, filename: str, schema: list[str], llm_repair: bool) -> OCRResponse:
        """Run OCR + entity extraction on one document."""
        try:
            r = await self._http.post(
                "/extract",
                files={"file": (filename, data)},
                data={"schema": ",".join(schema), "llm_repair": str(llm_repair).lower()},
            )
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise UpstreamUnavailable(f"OCR service error: {exc}") from exc
        return OCRResponse.model_validate(r.json())
