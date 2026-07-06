# Multimodal AI Platform

Four multimodal capabilities behind one production-patterned API: **image captioning**, **visual question answering**, **multimodal semantic search**, and **OCR + document intelligence**.

One Qwen3-VL-4B base model serves both captioning and VQA via runtime LoRA adapter switching in vLLM. SigLIP-2 + Qdrant power retrieval. PaddleOCR-VL + GLiNER power document extraction. A FastAPI gateway is the only public surface.

```
Frontend (Streamlit) → Gateway (FastAPI :8000)
                        ├── vLLM  (Qwen3-VL-4B + caption-lora + vqa-lora)
                        ├── Embeddings (SigLIP-2) ── Qdrant
                        └── OCR (PaddleOCR-VL + GLiNER)   ← isolated env
```

## Quickstart

```bash
cp .env.example .env            # set real keys
uv sync --all-packages          # dev environment
make test                       # unit tests (no GPU needed)
make up                         # dev profile: gateway + embeddings + qdrant
make full                       # everything, GPU required
make smoke                      # end-to-end smoke test
```

## Train the adapters (one GPU rule: never while vLLM is up)

```bash
uv run python data/scripts/download.py coco2017_train_images coco_karpathy_split
make train-caption              # QLoRA → artifacts/adapters/caption
make train-vqa                  # QLoRA → artifacts/adapters/vqa
```

## Results

| Task | Metric | Baseline (scratch) | Qwen3-VL zero-shot | + LoRA |
|---|---|---|---|---|
| Captioning (COCO Karpathy test) | CIDEr | _fill in_ | _fill in_ | _fill in_ |
| VQA v2 (val) | Official acc. | — | _fill in_ | _fill in_ |
| Retrieval (COCO 5k, t→i) | R@1 / R@5 | — | _fill in_ | — |
| OCR (SROIE) | Field F1 | — | _fill in_ | — |

Numbers are produced by the eval scripts in `packages/training/` and must reproduce within tolerance of published baselines before being claimed (see `docs/runbook.md`).

## Repository map

- `packages/gateway` — public API: validation, auth, rate limits, metrics, problem+json errors, circuit breaker
- `packages/embeddings` — SigLIP-2 service
- `packages/ocr` — PaddleOCR-VL + GLiNER (own environment on purpose — ADR-004)
- `packages/training` — QLoRA scripts, from-scratch baseline, official VQA metric, retrieval eval
- `docs/adr/` — every model and infrastructure decision, with the rejected alternative

## License / data

Code: MIT. Datasets keep their original licenses (see `data/manifest.yaml`); this repository is for research and portfolio use.
