# Runbook

## Start / stop
- Dev (no GPU): `make up` — gateway, embeddings (CPU fallback), qdrant
- Full: `make full` — adds vLLM, OCR, frontend, monitoring
- Stop everything: `make down`

## The one-GPU rule
vLLM reserves 85% of VRAM. Training while it runs will OOM.
`make train-caption` / `make train-vqa` refuse to start if vLLM is on the GPU.

## Promote an adapter
1. `make eval-gate` (new adapter must beat the previous on the frozen 500-example slice)
2. Write/update the model card in docs/model-cards/
3. Copy adapter to artifacts/adapters/<task>/ and restart the vllm service

## Backups
Qdrant: `curl -X POST http://localhost:6333/collections/images_siglip2_v1/snapshots`

## Known constraints
- SigLIP-2 text tower truncates at 64 tokens; long queries lose tail content.
- vLLM image tag is pinned; verify multimodal LoRA before bumping (ADR-005 fallback: merged checkpoints).
- OCR service lazy-loads models; first request after start is slow (~30-60 s).
