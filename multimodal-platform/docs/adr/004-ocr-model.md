# ADR-004: PaddleOCR-VL + GLiNER; isolated environment

**Status:** accepted

**Decision:** PaddleOCR-VL (0.9B) for layout-aware OCR — 109 languages including
Arabic, Apache 2.0, state-of-the-art at its size on document benchmarks. GLiNER for
zero-shot field extraction against a request-supplied schema. Optional `llm_repair`
pass sends missed fields to the shared VLM with a JSON schema, validated by Pydantic.

**Alternatives rejected:** DeepSeek-OCR — wins on throughput/cost at bulk scale
(MoE, 570M active); revisit if pages/sec becomes the bottleneck. EasyOCR — outclassed
for document intelligence in 2026.

**Hard rule:** PaddlePaddle never shares an environment or container with PyTorch.
The OCR package is outside the uv workspace and builds its own image.
