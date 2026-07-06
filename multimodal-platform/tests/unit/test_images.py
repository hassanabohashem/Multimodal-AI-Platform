"""Image validation rules that guard every upload endpoint."""
import io

import pytest
from PIL import Image

from mmp_common.images import MAX_SIDE, ImageValidationError, load_and_normalize


def _jpeg(size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "red").save(buf, format="JPEG")
    return buf.getvalue()


def test_valid_jpeg_roundtrip():
    img = load_and_normalize(_jpeg())
    assert img.mode == "RGB"


def test_oversize_rejected():
    with pytest.raises(ImageValidationError, match="exceeds"):
        load_and_normalize(b"x" * 100, max_bytes=10)


def test_garbage_rejected():
    with pytest.raises(ImageValidationError, match="decode"):
        load_and_normalize(b"not an image at all")


def test_large_image_downscaled():
    img = load_and_normalize(_jpeg((4000, 2000)))
    assert max(img.size) <= MAX_SIDE
