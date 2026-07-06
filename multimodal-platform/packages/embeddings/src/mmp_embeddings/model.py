"""SigLIP-2 encoder with simple dynamic batching.

Text tower note: SigLIP-2 text input caps at 64 tokens; longer queries are
truncated by the processor and the service returns a warning header.
"""
from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODEL_ID = "google/siglip2-so400m-patch16-384"


class SigLIP2Encoder:
    """Wraps the SigLIP-2 towers; all outputs are L2-normalized."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=dtype).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)

    @torch.inference_mode()
    def encode_text(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        inputs = self.processor(
            text=texts, padding="max_length", truncation=True, return_tensors="pt"
        ).to(self.device)
        feats = self.model.get_text_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.float().cpu().tolist()

    @torch.inference_mode()
    def encode_images(self, images: list[Image.Image]) -> list[list[float]]:
        """Embed a batch of PIL images."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.float().cpu().tolist()
