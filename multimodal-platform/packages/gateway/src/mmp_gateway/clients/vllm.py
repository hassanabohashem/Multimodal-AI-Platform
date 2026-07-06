"""Async client for the vLLM OpenAI-compatible server, with a circuit breaker.

The gateway only ever speaks the OpenAI chat API. If multimodal LoRA support
changes in a future vLLM release, we swap to merged checkpoints behind this
same interface (docs/adr/005-serving.md).
"""
from __future__ import annotations

import base64
import time

import httpx

from mmp_gateway.errors import UpstreamUnavailable


class CircuitBreaker:
    """Minimal half-open circuit breaker around one upstream."""

    def __init__(self, failures_to_open: int = 5, half_open_after_s: float = 20) -> None:
        self._threshold = failures_to_open
        self._cooldown = half_open_after_s
        self._failures = 0
        self._opened_at: float | None = None

    def before(self) -> None:
        """Raise fast if the circuit is open and still cooling down."""
        if self._opened_at is None:
            return
        if time.monotonic() - self._opened_at < self._cooldown:
            raise UpstreamUnavailable("circuit open: vLLM recently failing")
        self._opened_at = None  # half-open: allow one probe

    def record(self, ok: bool) -> None:
        """Update failure count; open the circuit past the threshold."""
        if ok:
            self._failures = 0
            return
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = time.monotonic()


class VLMClient:
    """Chat-completions client used by the caption and VQA routers."""

    def __init__(self, base_url: str, timeout_s: float) -> None:
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout_s)
        self._breaker = CircuitBreaker()

    async def aclose(self) -> None:
        """Close the connection pool."""
        await self._http.aclose()

    async def generate(self, model: str, prompt: str, image_jpeg: bytes, max_tokens: int = 64) -> str:
        """Run one image+text chat completion against the named (LoRA) model."""
        self._breaker.before()
        b64 = base64.b64encode(image_jpeg).decode()
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        }
        try:
            resp = await self._http.post("/chat/completions", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            self._breaker.record(ok=False)
            raise UpstreamUnavailable(f"vLLM error: {exc}") from exc
        self._breaker.record(ok=True)
        return str(resp.json()["choices"][0]["message"]["content"]).strip()
