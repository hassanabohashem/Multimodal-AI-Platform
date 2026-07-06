# ADR-001: One shared VLM with per-task LoRA adapters

**Status:** accepted

**Decision:** Serve Qwen3-VL-4B-Instruct once in vLLM; captioning and VQA are LoRA
adapters selected per request via the OpenAI `model` field. LoRA targets LLM-tower
projections only; the vision tower stays frozen so vLLM can serve the adapters.

**Alternatives rejected:** (a) two full fine-tuned checkpoints — 2x VRAM/disk on a
single-GPU budget; (b) Qwen3-VL-8B — better quality but does not train under QLoRA
in 12 GB VRAM; documented scale-up path.

**Consequences:** halves memory; adapter swap is per-request; if a vLLM release
regresses multimodal LoRA, fall back to merged checkpoints behind the same API (ADR-005).
