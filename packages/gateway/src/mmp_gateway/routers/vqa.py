"""POST /v1/vqa — visual question answering via the VQA LoRA adapter."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from mmp_common.images import load_and_normalize, to_jpeg_bytes
from mmp_common.schemas import VQAResponse
from mmp_gateway.middleware.auth import require_api_key
from mmp_gateway.settings import settings

router = APIRouter(tags=["vqa"])

_YESNO_PREFIXES = ("is ", "are ", "was ", "were ", "does ", "do ", "did ", "has ", "have ", "can ")
_NUMBER_PREFIX = "how many"


def classify_question(question: str) -> str:
    """Cheap answer-type heuristic mirroring the VQA v2 categories."""
    q = question.strip().lower()
    if q.startswith(_NUMBER_PREFIX):
        return "number"
    if q.startswith(_YESNO_PREFIXES):
        return "yes/no"
    return "other"


@router.post("/v1/vqa", response_model=VQAResponse, dependencies=[Depends(require_api_key)])
async def vqa(
    request: Request,
    image: UploadFile = File(...),
    question: str = Form(..., min_length=1, max_length=512),
    verbose: bool = Form(False),
) -> VQAResponse:
    """Answer a natural-language question about one uploaded image.

    verbose=true routes to the base model (no adapter) for conversational
    answers; the default routes to the benchmark-style short-answer adapter.
    """
    start = time.perf_counter()
    img = load_and_normalize(await image.read(), settings.max_image_mb * 1024 * 1024)
    model = settings.base_model if verbose else settings.vqa_model
    prompt = question if verbose else f"Answer with a short answer. Question: {question}"
    answer = await request.app.state.vlm.generate(
        model=model, prompt=prompt, image_jpeg=to_jpeg_bytes(img), max_tokens=96 if verbose else 16
    )
    return VQAResponse(
        request_id=request.state.request_id,
        answer=answer,
        answer_type=classify_question(question),  # type: ignore[arg-type]
        model_version=model,
        latency_ms=(time.perf_counter() - start) * 1000,
    )
