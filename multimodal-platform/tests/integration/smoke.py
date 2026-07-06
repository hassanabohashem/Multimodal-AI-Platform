"""End-to-end smoke test against a running `full` compose stack.

Usage: make full && uv run python tests/integration/smoke.py
Exits non-zero if any endpoint fails or misses its latency budget.
"""
from __future__ import annotations

import io
import os
import sys
import time

import httpx
from PIL import Image

GATEWAY = os.environ.get("MMP_GATEWAY_URL", "http://localhost:8000")
KEY = {"X-API-Key": os.environ.get("MMP_API_KEY", "change-me-demo-key")}
BUDGETS_S = {"caption": 8.0, "vqa": 6.0, "search": 1.0}


def _img() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (320, 240), "blue").save(buf, format="JPEG")
    return buf.getvalue()


def check(name: str, fn) -> None:
    start = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - start
    budget = BUDGETS_S.get(name)
    status = "OK" if budget is None or elapsed <= budget else "SLOW"
    print(f"[{status}] {name}: {elapsed:.2f}s")
    if status == "SLOW":
        sys.exit(1)


def main() -> None:
    check("health", lambda: httpx.get(f"{GATEWAY}/healthz").raise_for_status())
    check("caption", lambda: httpx.post(f"{GATEWAY}/v1/caption", headers=KEY, timeout=60,
                                        files={"image": ("i.jpg", _img())}).raise_for_status())
    check("vqa", lambda: httpx.post(f"{GATEWAY}/v1/vqa", headers=KEY, timeout=60,
                                    files={"image": ("i.jpg", _img())},
                                    data={"question": "What color is this?"}).raise_for_status())
    check("search", lambda: httpx.post(f"{GATEWAY}/v1/search", timeout=30,
                                       json={"query": "a blue square"}).raise_for_status())
    print("smoke passed")


if __name__ == "__main__":
    main()
