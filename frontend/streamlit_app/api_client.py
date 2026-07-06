"""Typed client for the gateway used by the Streamlit UI."""
from __future__ import annotations

import os
from typing import Any

import httpx

GATEWAY = os.environ.get("MMP_GATEWAY_URL", "http://gateway:8000")
API_KEY = os.environ.get("MMP_API_KEY", "change-me-demo-key")
_headers = {"X-API-Key": API_KEY}


def caption(image_bytes: bytes, style: str = "concise") -> dict[str, Any]:
    """Call POST /v1/caption."""
    r = httpx.post(f"{GATEWAY}/v1/caption", headers=_headers, timeout=60,
                   files={"image": ("img.jpg", image_bytes)}, data={"style": style})
    r.raise_for_status()
    return r.json()


def vqa(image_bytes: bytes, question: str, verbose: bool = False) -> dict[str, Any]:
    """Call POST /v1/vqa."""
    r = httpx.post(f"{GATEWAY}/v1/vqa", headers=_headers, timeout=60,
                   files={"image": ("img.jpg", image_bytes)},
                   data={"question": question, "verbose": str(verbose).lower()})
    r.raise_for_status()
    return r.json()


def search(query: str, top_k: int = 12) -> dict[str, Any]:
    """Call POST /v1/search."""
    r = httpx.post(f"{GATEWAY}/v1/search", timeout=30, json={"query": query, "top_k": top_k})
    r.raise_for_status()
    return r.json()


def ocr(file_bytes: bytes, filename: str, schema: str) -> dict[str, Any]:
    """Call POST /v1/ocr."""
    r = httpx.post(f"{GATEWAY}/v1/ocr", headers=_headers, timeout=180,
                   files={"file": (filename, file_bytes)}, data={"schema": schema})
    r.raise_for_status()
    return r.json()
