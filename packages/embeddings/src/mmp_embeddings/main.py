"""FastAPI wrapper around the SigLIP-2 encoder."""
from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

from mmp_common.logging import configure_logging
from mmp_embeddings.model import SigLIP2Encoder

log = configure_logging("embeddings")
INTERNAL_TOKEN = os.environ.get("MMP_INTERNAL_TOKEN", "change-me-internal-token")


async def require_internal(x_internal_token: str = Header(default="")) -> None:
    """Only the gateway may call this service."""
    if x_internal_token != INTERNAL_TOKEN:
        raise HTTPException(401, "internal token required")


class TextBatch(BaseModel):
    """Batch of texts to embed."""

    texts: list[str] = Field(min_length=1, max_length=64)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once; warm it so /readyz reflects real readiness."""
    app.state.encoder = SigLIP2Encoder()
    app.state.encoder.encode_text(["warmup"])
    log.info("embeddings ready")
    yield


app = FastAPI(title="mmp-embeddings", lifespan=lifespan)


@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict[str, str]:
    """Liveness."""
    return {"status": "ok"}


@app.get("/readyz", include_in_schema=False)
async def readyz() -> dict[str, str]:
    """Readiness: model is loaded and warm."""
    return {"status": "ready"}


@app.post("/embed/text", dependencies=[Depends(require_internal)])
async def embed_text(batch: TextBatch) -> dict[str, list[list[float]]]:
    """Embed up to 64 texts."""
    return {"embeddings": app.state.encoder.encode_text(batch.texts)}


@app.post("/embed/image", dependencies=[Depends(require_internal)])
async def embed_image(file: UploadFile = File(...)) -> dict[str, list[list[float]]]:
    """Embed one image."""
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    return {"embeddings": app.state.encoder.encode_images([img])}
