"""POST /v1/search — multimodal semantic search over the Qdrant index."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, File, Request, UploadFile

from mmp_common.images import load_and_normalize, to_jpeg_bytes
from mmp_common.schemas import SearchRequest, SearchResponse
from mmp_gateway.middleware.auth import require_api_key

router = APIRouter(tags=["search"])


@router.post("/v1/search", response_model=SearchResponse)
async def search_text(request: Request, body: SearchRequest) -> SearchResponse:
    """Text→image retrieval."""
    start = time.perf_counter()
    vector = await request.app.state.embeddings.embed_text(body.query)
    hits = await request.app.state.store.search(vector, body.top_k, body.filters)
    return SearchResponse(
        request_id=request.state.request_id,
        hits=hits,
        latency_ms=(time.perf_counter() - start) * 1000,
    )


@router.post("/v1/search/by-image", response_model=SearchResponse, dependencies=[Depends(require_api_key)])
async def search_image(request: Request, image: UploadFile = File(...), top_k: int = 10) -> SearchResponse:
    """Image→image retrieval."""
    start = time.perf_counter()
    img = load_and_normalize(await image.read())
    vector = await request.app.state.embeddings.embed_image(to_jpeg_bytes(img))
    hits = await request.app.state.store.search(vector, min(top_k, 50), None)
    return SearchResponse(
        request_id=request.state.request_id,
        hits=hits,
        latency_ms=(time.perf_counter() - start) * 1000,
    )
