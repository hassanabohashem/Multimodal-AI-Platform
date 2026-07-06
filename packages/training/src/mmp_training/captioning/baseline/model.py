"""From-scratch captioning baseline: ViT-S encoder -> Transformer decoder.

Demonstrates fundamentals: patch embeddings, causal masking, cross-attention,
teacher forcing, and beam-search decoding — with modern components.
"""
from __future__ import annotations

import math

import timm
import torch
from torch import Tensor, nn


class CaptionDecoder(nn.Module):
    """Causal Transformer decoder that cross-attends to ViT patch tokens."""

    def __init__(self, vocab_size: int, d_model: int = 512, heads: int = 8,
                 layers: int = 6, ff: int = 2048, dropout: float = 0.1, max_len: int = 64) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerDecoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, tokens: Tensor, memory: Tensor) -> Tensor:
        """Next-token logits given target tokens and encoder memory."""
        seq_len = tokens.size(1)
        x = self.token_emb(tokens) * math.sqrt(self.d_model) + self.pos_emb[:, :seq_len]
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=tokens.device)
        out = self.decoder(x, memory, tgt_mask=mask, tgt_is_causal=True)
        return self.head(out)


class BaselineCaptioner(nn.Module):
    """ViT-S/16 encoder (timm) + CaptionDecoder."""

    def __init__(self, vocab_size: int, encoder_name: str = "vit_small_patch16_224",
                 d_model: int = 512, freeze_encoder: bool = True) -> None:
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=True, num_classes=0)
        enc_dim = self.encoder.embed_dim
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.proj = nn.Linear(enc_dim, d_model)
        self.decoder = CaptionDecoder(vocab_size, d_model=d_model)

    def forward(self, images: Tensor, tokens: Tensor) -> Tensor:
        """Teacher-forced logits for a batch of (image, caption) pairs."""
        patches = self.encoder.forward_features(images)  # (B, N, enc_dim)
        memory = self.proj(patches)
        return self.decoder(tokens, memory)
