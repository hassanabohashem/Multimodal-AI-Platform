"""POST /v1/caption — image captioning via the caption LoRA adapter."""
from __future__ import annotations

import time
from typing import Literal

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from mmp_common.images import load_and_normalize, to_jpeg_bytes
from mmp_common.schemas import CaptionResponse
from mmp_gateway.middleware.auth import require_api_key
from mmp_gateway.settings import settings

router = APIRouter(tags=["caption"])

_PROMPTS = {
    "concise": "Describe this image in one concise sentence.",
    "detailed": "Describe this image in detail, covering objects, actions, and setting.",
}


@router.post("/v1/caption", response_model=CaptionResponse, dependencies=[Depends(require_api_key)])
async def caption(
    request: Request,
    image: UploadFile = File(...),
    style: Literal["concise", "detailed"] = Form("concise"),
    max_tokens: int = Form(64, ge=8, le=256),
) -> CaptionResponse:
    """Generate a caption for one uploaded image."""
    start = time.perf_counter()
    img = load_and_normalize(await image.read(), settings.max_image_mb * 1024 * 1024)
    text = await request.app.state.vlm.generate(
        model=settings.caption_model,
        prompt=_PROMPTS[style],
        image_jpeg=to_jpeg_bytes(img),
        max_tokens=max_tokens,
    )
    return CaptionResponse(
        request_id=request.state.request_id,
        caption=text,
        model_version=settings.caption_model,
        latency_ms=(time.perf_counter() - start) * 1000,
    )
