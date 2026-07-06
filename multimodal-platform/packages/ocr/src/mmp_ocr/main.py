"""FastAPI wrapper around the document pipeline."""
from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile

from mmp_ocr.pipeline import DocumentPipeline, rasterize

INTERNAL_TOKEN = os.environ.get("MMP_INTERNAL_TOKEN", "change-me-internal-token")


async def require_internal(x_internal_token: str = Header(default="")) -> None:
    """Only the gateway may call this service."""
    if x_internal_token != INTERNAL_TOKEN:
        raise HTTPException(401, "internal token required")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Construct the pipeline (models lazy-load on first request)."""
    app.state.pipeline = DocumentPipeline()
    yield


app = FastAPI(title="mmp-ocr", lifespan=lifespan)


@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict[str, str]:
    """Liveness."""
    return {"status": "ok"}


@app.get("/readyz", include_in_schema=False)
async def readyz() -> dict[str, str]:
    """Readiness."""
    return {"status": "ready"}


@app.post("/extract", dependencies=[Depends(require_internal)])
async def extract(
    file: UploadFile = File(...),
    schema: str = Form(""),
    llm_repair: bool = Form(False),
) -> dict:
    """OCR one document and extract the requested fields."""
    start = time.perf_counter()
    warnings: list[str] = []
    fields = [s.strip() for s in schema.split(",") if s.strip()]
    try:
        images = rasterize(await file.read(), file.filename or "upload")
    except ValueError as exc:
        raise HTTPException(413, str(exc)) from exc
    pages = app.state.pipeline.extract_pages(images)
    full_text = "\n\n".join(p.markdown for p in pages)
    entities = app.state.pipeline.extract_entities(full_text, fields)
    missing = [f for f, v in entities.items() if v["value"] is None]
    if missing:
        warnings.append(f"fields not found by NER: {missing}")
        if llm_repair:
            warnings.append("llm_repair requested: wire the vLLM JSON pass here (see docs/adr/004)")
    return {
        "request_id": str(uuid.uuid4()),
        "pages": [{"markdown": p.markdown, "blocks": p.blocks} for p in pages],
        "entities": entities,
        "warnings": warnings,
        "latency_ms": (time.perf_counter() - start) * 1000,
    }
