"""API contracts shared by the gateway and internal services.

These models are the single source of truth: the gateway validates against
them and every service imports them, so contracts cannot drift.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    """Base model that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid")


# ---------- captioning ----------
class CaptionResponse(Strict):
    """Caption result for a single image."""

    request_id: str
    caption: str
    model_version: str
    latency_ms: float


# ---------- VQA ----------
class VQARequest(Strict):
    """Question about an uploaded image (image travels as multipart)."""

    question: str = Field(min_length=1, max_length=512)
    verbose: bool = False


class VQAResponse(Strict):
    """Answer to a visual question."""

    request_id: str
    answer: str
    answer_type: Literal["yes/no", "number", "other"]
    model_version: str
    latency_ms: float


# ---------- search ----------
class SearchFilters(Strict):
    """Optional payload filters applied in Qdrant."""

    tags: list[str] | None = None
    source: str | None = None


class SearchRequest(Strict):
    """Text search request (image-as-query uses multipart on the same route)."""

    query: str = Field(min_length=1, max_length=256)
    top_k: int = Field(default=10, ge=1, le=50)
    filters: SearchFilters | None = None


class SearchHit(Strict):
    """One retrieved image."""

    image_id: str
    score: float
    uri: str
    thumb_uri: str | None = None
    caption: str | None = None


class SearchResponse(Strict):
    """Ranked retrieval results."""

    request_id: str
    hits: list[SearchHit]
    latency_ms: float


class IndexRequest(Strict):
    """Batch ingestion request."""

    items: list[IndexItem] = Field(min_length=1, max_length=256)


class IndexItem(Strict):
    """One image to ingest into the search index."""

    uri: str
    caption: str | None = None
    tags: list[str] = Field(default_factory=list)
    source: str = "user"


class IndexJob(Strict):
    """Handle for an asynchronous ingestion job."""

    job_id: str
    status: Literal["queued", "running", "done", "failed"]
    total: int
    done: int = 0
    error: str | None = None


# ---------- OCR ----------
class OCRBlock(Strict):
    """One layout block detected on a page."""

    text: str
    bbox: tuple[float, float, float, float]
    type: Literal["text", "table", "figure", "title"]


class OCRPage(Strict):
    """Extraction result for a single page."""

    markdown: str
    blocks: list[OCRBlock]


class OCREntity(Strict):
    """A structured field extracted from the document."""

    value: str | None
    confidence: float | None = None
    source_bbox: tuple[float, float, float, float] | None = None


class OCRResponse(Strict):
    """Full document-intelligence result."""

    request_id: str
    pages: list[OCRPage]
    entities: dict[str, OCREntity]
    warnings: list[str] = Field(default_factory=list)
    latency_ms: float


class Problem(Strict):
    """RFC 9457 problem+json error body."""

    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None
    request_id: str | None = None


IndexRequest.model_rebuild()
