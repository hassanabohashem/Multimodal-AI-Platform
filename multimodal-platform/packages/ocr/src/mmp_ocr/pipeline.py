"""Document extraction pipeline: rasterize -> PaddleOCR-VL -> GLiNER.

PaddleOCR-VL (0.9B) handles 109 languages including Arabic and produces
layout-aware output. GLiNER performs zero-shot NER against a request-supplied
schema, so no field-extraction training is required.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import pypdfium2 as pdfium
from PIL import Image

MAX_PDF_PAGES = 20


@dataclass
class PageResult:
    """OCR result for one page."""

    markdown: str
    blocks: list[dict[str, Any]] = field(default_factory=list)


def rasterize(data: bytes, filename: str) -> list[Image.Image]:
    """Turn an upload (image or PDF) into a list of RGB page images at ~200 dpi."""
    if filename.lower().endswith(".pdf"):
        doc = pdfium.PdfDocument(io.BytesIO(data))
        if len(doc) > MAX_PDF_PAGES:
            raise ValueError(f"PDF exceeds {MAX_PDF_PAGES} pages")
        return [page.render(scale=200 / 72).to_pil().convert("RGB") for page in doc]
    return [Image.open(io.BytesIO(data)).convert("RGB")]


class DocumentPipeline:
    """Lazy-loaded PaddleOCR-VL + GLiNER pipeline."""

    def __init__(self) -> None:
        self._ocr: Any = None
        self._ner: Any = None

    def _ensure_loaded(self) -> None:
        if self._ocr is None:
            from paddleocr import PaddleOCRVL  # heavy import deferred to first use

            self._ocr = PaddleOCRVL()
        if self._ner is None:
            from gliner import GLiNER

            self._ner = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

    def extract_pages(self, images: list[Image.Image]) -> list[PageResult]:
        """Run layout-aware OCR on each page."""
        self._ensure_loaded()
        results: list[PageResult] = []
        for img in images:
            out = self._ocr.predict(img)
            page = out[0] if isinstance(out, list) else out
            markdown = getattr(page, "markdown", None) or page.get("markdown", "")
            blocks = []
            for b in page.get("layout_blocks", page.get("blocks", [])):
                blocks.append({
                    "text": b.get("text", ""),
                    "bbox": tuple(b.get("bbox", (0, 0, 0, 0))),
                    "type": b.get("type", "text"),
                })
            results.append(PageResult(markdown=str(markdown), blocks=blocks))
        return results

    def extract_entities(self, text: str, schema: list[str]) -> dict[str, dict[str, Any]]:
        """Zero-shot NER over the concatenated page text."""
        self._ensure_loaded()
        entities: dict[str, dict[str, Any]] = {f: {"value": None, "confidence": None} for f in schema}
        if not schema or not text.strip():
            return entities
        for ent in self._ner.predict_entities(text, schema, threshold=0.4):
            label = ent["label"]
            best = entities.get(label, {})
            if best.get("confidence") is None or ent["score"] > best["confidence"]:
                entities[label] = {"value": ent["text"], "confidence": round(float(ent["score"]), 4)}
        return entities
