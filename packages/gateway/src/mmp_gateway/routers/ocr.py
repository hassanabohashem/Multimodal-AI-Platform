"""POST /v1/ocr — document intelligence via the OCR service."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from mmp_common.schemas import OCRResponse
from mmp_gateway.middleware.auth import require_api_key

router = APIRouter(tags=["ocr"])


@router.post("/v1/ocr", response_model=OCRResponse, dependencies=[Depends(require_api_key)])
async def ocr(
    request: Request,
    file: UploadFile = File(...),
    extract_fields: str = Form("", alias="schema", description="Comma-separated field names, e.g. date,total_amount,vendor"),
    llm_repair: bool = Form(False),
) -> OCRResponse:
    """Extract layout-aware text and structured fields from an image or PDF."""
    fields = [s.strip() for s in extract_fields.split(",") if s.strip()]
    result = await request.app.state.ocr.extract(
        await file.read(), file.filename or "upload", fields, llm_repair
    )
    result.request_id = request.state.request_id
    return result
