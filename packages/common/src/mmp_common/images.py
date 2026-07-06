"""Image validation and normalization used by the gateway and training code."""
from __future__ import annotations

import io

from PIL import Image, ImageOps

MAX_SIDE = 1536
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class ImageValidationError(ValueError):
    """Raised when an upload fails validation; maps to HTTP 415/422."""


def load_and_normalize(data: bytes, max_bytes: int = 10 * 1024 * 1024) -> Image.Image:
    """Decode, validate, EXIF-correct, convert to RGB, and cap the long side.

    Args:
        data: Raw upload bytes.
        max_bytes: Size ceiling before decoding is attempted.

    Returns:
        A normalized RGB PIL image, longest side <= MAX_SIDE.

    Raises:
        ImageValidationError: On oversize, undecodable, or disallowed formats.
    """
    if len(data) > max_bytes:
        raise ImageValidationError(f"image exceeds {max_bytes} bytes")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # noqa: BLE001 - PIL raises many types
        raise ImageValidationError("could not decode image") from exc
    if (img.format or "").upper() not in ALLOWED_FORMATS:
        raise ImageValidationError(f"unsupported format: {img.format}")
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    """Re-encode a normalized image as JPEG (strips metadata)."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
