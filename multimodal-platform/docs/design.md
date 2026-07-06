# Multimodal AI Platform — Production Design Document

**Version:** 1.0 · **Date:** July 2026 · **Status:** Ready for implementation
**Target:** Portfolio-grade, production-patterned multimodal platform (captioning, VQA, semantic search, OCR/document intelligence)

---

## 1. Executive Summary

This document specifies a complete, implementable design for a four-module multimodal AI platform. Every model, library, and infrastructure choice below was checked against the mid-2026 landscape for maintenance status, license, mutual compatibility, and fit on constrained hardware (baseline assumption: **one consumer GPU, 12–16 GB VRAM**, with notes for Colab/Kaggle fallback).

The single most important architectural decision, and the one that makes this feel like a real product rather than four notebooks: **one shared VLM serves both captioning and VQA**, deployed once behind an OpenAI-compatible inference server (vLLM), with LoRA adapters swapped per task. Search and OCR run as separate lightweight services. A FastAPI gateway is the only public surface. Everything ships via Docker Compose with CI/CD, experiment tracking, and monitoring.

### Verified technology stack (as of July 2026)

| Layer | Choice | Why this and not the alternative |
|---|---|---|
| Language | Python 3.12 | 3.12 is the sweet spot: full ecosystem support (PyTorch, vLLM, Paddle all ship 3.12 wheels); 3.13 still has gaps in some CUDA-adjacent packages. |
| Package mgmt | **uv** (Astral) | Replaced Poetry/pip as the de facto standard in 2025–26. Lockfile (`uv.lock`) gives reproducibility; 10–100x faster resolution; single tool for venvs + deps. |
| DL framework | PyTorch 2.7+ (CUDA 12.x) | Industry default; `torch.compile` for the baseline model; required by every model below. |
| VLM (captioning + VQA) | **Qwen3-VL-4B-Instruct** (Apache 2.0), QLoRA fine-tuned | Current-generation successor to Qwen2.5-VL (released Oct 2025, dense 2B/4B/8B/32B variants). Strong OCR/doc/VQA performance, multilingual incl. Arabic, first-class support in Transformers, PEFT, and vLLM. 4B fits QLoRA training in ~11 GB VRAM. Use the 8B variant if you have 24 GB. |
| Baseline (from scratch) | ViT-small encoder + 6-layer Transformer decoder (option B: ResNet50+LSTM w/ attention) | The brief said CNN+LSTM; keep it as option B, but a small ViT→Transformer decoder is the more interview-relevant "fundamentals" demo in 2026 and is barely harder to write. |
| Embeddings (search) | **SigLIP 2** `google/siglip2-so400m-patch16-384` (Apache 2.0) | Strongest open image–text similarity model as of mid-2026; sigmoid loss → well-calibrated scores; permissive license. **Note:** jina-embeddings-v4 wins on visually rich documents and multilingual queries but is CC-BY-NC (non-commercial) — fine to mention in the README as an evaluated alternative, wrong choice for a "production" story. |
| Vector store | **Qdrant** (self-hosted Docker) | Production-grade: payload filtering, HNSW, snapshots, gRPC + REST, scalar/binary quantization. FAISS stays as an in-process index inside evaluation notebooks only. |
| OCR | **PaddleOCR-VL** (0.9B) primary; DeepSeek-OCR as the high-throughput alternative | PaddleOCR-VL covers 109 languages including Arabic, tops OmniDocBench-class layout benchmarks at sub-1B size, runs on modest GPUs. DeepSeek-OCR (3B MoE, 570M active) wins on throughput/token-efficiency for bulk pipelines. EasyOCR is no longer competitive for document work — dropped. |
| Structured extraction (NER) | **GLiNER** (zero-shot NER) + Pydantic-validated VLM JSON extraction as fallback | GLiNER gives schema-free entity extraction with no training; for receipts/forms, prompting the shared Qwen3-VL with a JSON schema and validating with Pydantic is the modern pattern. |
| Inference serving | **vLLM** (OpenAI-compatible server) for the VLM; FastAPI-embedded for SigLIP-2 and OCR | vLLM natively serves Qwen3-VL with multi-image input and **runtime LoRA adapter loading** — one base model, two adapters (caption, vqa). This is the pattern real companies use. |
| API gateway | FastAPI + Pydantic v2, fully async | Gateway holds no model weights; it validates, routes, rate-limits, logs, and calls internal services over HTTP/gRPC. |
| Frontend | Phase 1: **Streamlit**; Phase 2 (stretch): React 18 + Vite + Tailwind | Streamlit gets you a demo in days and talks to the same API. The React app is a polish stretch goal, not a dependency of anything. |
| Experiment tracking | Weights & Biases (free tier) | Runs, sweeps, model artifacts, eval tables. All training scripts log to W&B; best checkpoints pushed to the Hugging Face Hub as the "model registry." |
| Data versioning | Hugging Face Datasets + a `data/manifest.yaml` with SHA256s | Full DVC is overkill for a solo project; a checksummed manifest + deterministic preprocessing scripts gives you the reproducibility story without the tooling tax. |
| Containers | Docker + Docker Compose (profiles: `dev`, `full`, `monitoring`) | Single `docker compose --profile full up` brings up gateway, vLLM, OCR, embeddings, Qdrant, frontend, Prometheus, Grafana. |
| CI/CD | GitHub Actions: lint → typecheck → test → build → (on tag) push images | Ruff (lint+format), mypy (strict on `src/`), pytest with coverage gate, Docker buildx with layer caching. |
| Monitoring | Prometheus + Grafana + structlog JSON logs | Request latency histograms per endpoint, GPU memory gauge, error rates, and a basic embedding-drift check (mean cosine distance of daily query embeddings vs. reference). |
| Code quality | Ruff, mypy (strict), pre-commit, Google-style docstrings | Ruff replaces black/isort/flake8 in one tool. |

### Compatibility review (the conflicts I checked so you don't hit them)

1. **vLLM ↔ Qwen3-VL ↔ LoRA:** vLLM supports Qwen3-VL with image input and LoRA adapters on the *language* tower. Therefore: **train LoRA on LLM layers only** (`target_modules` = attention + MLP projections of the decoder), keep the vision tower frozen. Merging adapters is the escape hatch if a vLLM version regresses multimodal-LoRA support — the design works either way because the gateway only speaks the OpenAI API.
2. **PaddlePaddle ↔ PyTorch in one process:** don't. PaddleOCR-VL runs in its **own container** with its own environment. This is why the OCR module is a separate service, not a library import in the gateway.
3. **bitsandbytes 4-bit ↔ 12 GB VRAM:** Qwen3-VL-4B in NF4 with gradient checkpointing and batch size 1–2 (grad accumulation 8–16) trains within ~11 GB at 1024-token sequences. The 8B variant does **not** reliably fit training in 12 GB; it needs 24 GB. The design defaults to 4B.
4. **SigLIP 2 in Transformers:** supported since Transformers 4.49 (Feb 2025); use `AutoModel` + `AutoProcessor`, normalize embeddings, cosine distance in Qdrant. Text encoder max length is 64 tokens — the gateway truncates and warns; long queries get summarized client-side.
5. **Training vs. serving GPU contention:** you have one GPU. Compose profiles ensure the vLLM service is **down** while training runs. Documented in the runbook; enforced by a `make train` target that checks `nvidia-smi` for the vLLM process.
6. **Licenses:** Qwen3-VL (Apache 2.0), SigLIP 2 (Apache 2.0), PaddleOCR-VL (Apache 2.0), GLiNER (Apache 2.0), Qdrant (Apache 2.0), vLLM (Apache 2.0). Datasets: MS-COCO (CC-BY 4.0 annotations), VQA v2 (CC-BY), SROIE/FUNSD (research use — note in README). No copyleft or non-commercial surprises in the core path.

---

## 2. System Architecture

### 2.1 Service topology

```
                        ┌────────────────────────────────────────────────┐
                        │                 docker network                 │
                        │                                                │
 ┌──────────┐   HTTPS   │  ┌─────────────┐        ┌────────────────────┐ │
 │ Frontend │──────────▶│  │  Gateway    │──HTTP──▶ vLLM (Qwen3-VL-4B) │ │
 │ Streamlit│           │  │  FastAPI    │        │ + LoRA: caption,vqa│ │
 └──────────┘           │  │  :8000      │        │ :8001 (OpenAI API) │ │
                        │  │             │        └────────────────────┘ │
                        │  │  - auth     │        ┌────────────────────┐ │
                        │  │  - validate │──HTTP──▶ Embedding service  │ │
                        │  │  - route    │        │ SigLIP-2, FastAPI  │ │
                        │  │  - metrics  │        │ :8002              │ │
                        │  │  - logging  │        └───────┬────────────┘ │
                        │  │             │                │ upsert/query │
                        │  │             │        ┌───────▼────────────┐ │
                        │  │             │──gRPC──▶ Qdrant :6333/6334  │ │
                        │  │             │        └────────────────────┘ │
                        │  │             │        ┌────────────────────┐ │
                        │  │             │──HTTP──▶ OCR service        │ │
                        │  └──────┬──────┘        │ PaddleOCR-VL+GLiNER│ │
                        │         │ /metrics      │ :8003              │ │
                        │  ┌──────▼──────┐        └────────────────────┘ │
                        │  │ Prometheus  │──▶ Grafana :3000              │
                        │  └─────────────┘                               │
                        └────────────────────────────────────────────────┘
```

Design rules that make this scalable and modular:

- **The gateway owns zero model weights.** It can restart in milliseconds, scale horizontally, and be tested without a GPU.
- **Every inference service exposes `/healthz` (liveness), `/readyz` (model loaded), and `/metrics`.** Compose `depends_on: condition: service_healthy` sequences startup correctly.
- **Services communicate over the internal Docker network only**; the gateway is the single published port. Internal service tokens (env-injected) prevent the frontend from bypassing the gateway.
- **Async end-to-end.** The gateway uses `httpx.AsyncClient` with connection pooling and per-service timeouts (captioning 30 s, VQA 30 s, search 5 s, OCR 120 s) and returns RFC 9457 `application/problem+json` errors.

### 2.2 Request flows

**Caption:** client → `POST /v1/caption` (multipart image) → gateway validates (type, ≤10 MB, decodable, EXIF-stripped, resized ≤1536 px) → base64 → vLLM chat completion with `model="caption-lora"` → postprocess → response with `request_id`, latency, model version.

**VQA:** same path, `model="vqa-lora"`, question injected into the chat template. Answer-type hint (yes/no, number, open) returned when the router detects it.

**Search:** text query → embedding service (SigLIP-2 text tower) → Qdrant `search` with optional payload filter (`tags`, `date`, `source`) → top-k ids + scores → gateway resolves thumbnail URLs. Image-as-query uses the vision tower; ingestion (`POST /v1/search/index`) is a background task with an idempotency key.

**OCR:** image/PDF page → OCR service → PaddleOCR-VL layout-aware extraction (markdown + boxes) → GLiNER entities with schema from the request (`["date","total_amount","vendor","invoice_number"]`) → optional VLM JSON-repair pass, Pydantic-validated → structured response.

### 2.3 Why one shared VLM (and where sharing stops)

Captioning and VQA are the same base model with different LoRA adapters, because (a) it halves VRAM and disk, (b) vLLM hot-loads adapters per request via the `model` field, and (c) "we consolidated two product features onto one served model with per-task adapters" is a genuinely strong systems-design line in an interview. Search does **not** share this backbone: dual-encoder retrieval needs a contrastively trained embedding space, and SigLIP 2 is purpose-built for that. OCR doesn't share it either: PaddleOCR-VL at 0.9B is an order of magnitude cheaper per page than pushing every document through a 4B generalist. Knowing where *not* to unify is part of the design.

---

## 3. Repository Structure

Monorepo, one `uv` workspace, one Docker Compose file, packages per service.

```
multimodal-platform/
├── pyproject.toml                 # uv workspace root
├── uv.lock
├── Makefile                       # make dev / train / test / up / down / lint
├── docker-compose.yml             # profiles: dev, full, monitoring
├── .github/workflows/
│   ├── ci.yml                     # lint → typecheck → test → build
│   └── release.yml                # on tag: build+push images, deploy
├── .pre-commit-config.yaml
├── configs/                       # hydra-style YAML, one per experiment
│   ├── caption_qlora.yaml
│   ├── vqa_qlora.yaml
│   ├── baseline_vit.yaml
│   └── serving.yaml
├── data/
│   ├── manifest.yaml              # dataset name → url, sha256, license
│   └── scripts/download.py        # deterministic fetch + verify
├── packages/
│   ├── common/                    # shared: schemas, logging, image utils
│   │   └── src/mmp_common/
│   ├── gateway/
│   │   └── src/mmp_gateway/
│   │       ├── main.py            # app factory, lifespan, middleware
│   │       ├── routers/           # caption.py, vqa.py, search.py, ocr.py
│   │       ├── clients/           # vllm.py, embeddings.py, ocr.py, qdrant.py
│   │       ├── middleware/        # request_id, timing, auth, rate_limit
│   │       └── settings.py        # pydantic-settings, env-driven
│   ├── embeddings/
│   │   └── src/mmp_embeddings/    # SigLIP-2 service + Qdrant ops
│   ├── ocr/
│   │   └── src/mmp_ocr/           # PaddleOCR-VL + GLiNER (own env!)
│   └── training/
│       └── src/mmp_training/
│           ├── captioning/        # baseline/ (from scratch), qlora/
│           ├── vqa/
│           ├── retrieval_eval/
│           └── callbacks.py       # W&B logging, sample predictions table
├── frontend/
│   ├── streamlit_app/
│   └── web/                       # (stretch) React + Vite + Tailwind
├── tests/
│   ├── unit/                      # no GPU, no network: schemas, utils, routing
│   ├── integration/               # against docker-compose dev profile
│   └── e2e/                       # smoke: full stack, tiny fixtures
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/dashboards/*.json
├── notebooks/                     # EDA + error analysis only; no logic lives here
└── docs/
    ├── architecture.md            # + mermaid diagram
    ├── runbook.md                 # start/stop, train-vs-serve GPU rule, backups
    ├── adr/                       # Architecture Decision Records (001-...)
    └── model-cards/               # one per fine-tuned adapter
```

Conventions enforced by CI: `src/` layout, type hints everywhere (`mypy --strict` on `packages/`), Google-style docstrings, no logic in notebooks, every model choice recorded as an ADR (this document seeds ADRs 001–006).

---

## 4. Module 1 — Image Captioning

### 4.1 Models

**Baseline (from scratch, demonstrates fundamentals).** Recommended: ViT-S/16 encoder (weights from `timm`, then unfrozen in stage 2) → learned linear projection → 6-layer Transformer decoder (d_model 512, 8 heads, cross-attention to patch tokens), trained with teacher forcing + label smoothing 0.1, beam search (k=3) at inference. Keep the brief's CNN+LSTM (ResNet-50 → attention LSTM, Show-Attend-Tell style) as option B if you want the classic; the ViT→Transformer version exercises the same fundamentals (tokenization, causal masking, cross-attention, decoding strategies) and reads as current. Tokenizer: train a 8k BPE on COCO captions with `tokenizers` — a small, honest from-scratch detail interviewers like.

**Production model.** Qwen3-VL-4B-Instruct + QLoRA. Fine-tuning a strong instruct VLM on COCO mostly teaches *style control* (concise, literal, one-sentence captions vs. its chatty default) and domain adaptation — say exactly that in the README; overstating what fine-tuning does here is a red flag to reviewers.

### 4.2 Data

- **MS-COCO 2017** (118k train / 5k val, 5 captions each) — primary. Karpathy split for comparability with published numbers.
- **Flickr30k** — held-out *transfer* evaluation only (never trained on). Reporting COCO-trained → Flickr30k zero-shot is a stronger portfolio signal than training on both.
- Preprocessing pipeline (shared package): decode → EXIF orientation fix → RGB → resize policy per model → augmentation (baseline only: RandomResizedCrop 0.8–1.0, horizontal flip — **flip disabled** for images whose captions contain left/right terms; catch this in a unit test).

### 4.3 Training pipeline (QLoRA)

```yaml
# configs/caption_qlora.yaml (key fields)
model: Qwen/Qwen3-VL-4B-Instruct
quant: {load_in_4bit: true, bnb_4bit_quant_type: nf4, bnb_4bit_compute_dtype: bfloat16}
lora:  {r: 16, alpha: 32, dropout: 0.05,
        target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]}  # LLM only; vision tower frozen (vLLM LoRA constraint)
train: {epochs: 1, lr: 1.0e-4, scheduler: cosine, warmup_ratio: 0.03,
        per_device_batch: 1, grad_accum: 16, bf16: true,
        gradient_checkpointing: true, max_seq_len: 1024, seed: 42}
data:  {dataset: coco_karpathy, subset: 60000, prompt_template: caption_v2}
log:   {wandb_project: mmp-caption, eval_every: 500, sample_table_every: 500}
```

Stack: `transformers` + `trl` `SFTTrainer` + `peft` + `bitsandbytes`. Loss masked to the assistant turn only. Every run logs to W&B: loss curves, LR, GPU memory, and a predictions table (image, reference, generated) every 500 steps — this table is what you screenshot for the README. Checkpoint = adapter only (~50 MB), pushed to HF Hub with a model card (data, config, metrics, limitations). Reproducibility: seed pinned, `uv.lock` committed, config hash logged to the run.

Compute reality check: 60k samples × 1 epoch at ~1.5 s/step (accumulated) ≈ 6–8 h on a 12 GB card, or two Colab Pro sessions with resume-from-checkpoint (checkpoints saved to Drive/HF every 500 steps).

### 4.4 Evaluation

`pycocoevalcap` for BLEU-4, METEOR, CIDEr, SPICE on the Karpathy test split, identical decoding config across models. Report a 3-row table: from-scratch baseline / Qwen3-VL zero-shot / Qwen3-VL + LoRA. Add qualitative error buckets (hallucinated objects, counting errors, missed relations) with 5 examples each — error analysis is worth more in interviews than a CIDEr decimal.

### 4.5 API

`POST /v1/caption` → multipart `image`, optional `style` (`concise|detailed`), `max_tokens`. Response: `{request_id, caption, model_version, latency_ms}`. Errors: 415 unsupported type, 413 too large, 422 undecodable, 503 model warming (with `Retry-After`).

---

## 5. Module 2 — Visual Question Answering

### 5.1 Model and data

Same served base model, second LoRA adapter (`vqa-lora`), trained on **VQA v2.0** (balanced pairs; ~443k train questions on COCO images — you already have the images). Sample 100–150k for the 12 GB budget; full-set training is a documented scale-up path. Prompt template instructs *short answers* to match the benchmark's answer style. Optional +20k **TextVQA** mix-in to strengthen text-in-image questions (synergy with Module 4).

### 5.2 Evaluation — the part that makes this module portfolio-grade

- Official **VQA accuracy metric**: `min(#humans_matching/3, 1)` over 10 human answers, with the official answer normalization (articles, punctuation, number words). Implement it as a tested utility; many public repos get normalization subtly wrong.
- **Breakdown by question type**: yes/no, number, other — plus a finer split by question prefix ("what color", "how many", "is there").
- **Error analysis notebook**: confusion patterns on counting questions, language-prior failures (answering from the question without the image — test this by re-running eval with a gray image; the gap you measure *is* the visual grounding). That gray-image ablation is an unusual, memorable result to put in the README.

### 5.3 API

`POST /v1/vqa` → image + `question` (1–512 chars). Response: `{request_id, answer, answer_type, model_version, latency_ms}`. Note in docs: answers are benchmark-style short; a `verbose=true` flag switches to the base model without the adapter for conversational answers — a nice demonstration of adapter routing.

---

## 6. Module 3 — Multimodal Semantic Search

### 6.1 Design

Dual-encoder retrieval with **SigLIP 2 (so400m/384)** producing 1152-d embeddings, L2-normalized, stored in **Qdrant** with cosine distance, HNSW (`m=16, ef_construct=128`), and scalar (int8) quantization enabled from day one — quantization is nearly free in recall here and lets you honestly write "4× memory reduction with <1% recall loss (measured)".

Collection schema:

```
collection: images_siglip2_v1        # version in the name → painless re-embedding migrations
vector: 1152, cosine
payload: {image_id, uri, thumb_uri, caption, tags[], source, ingested_at, width, height, sha256}
payload indexes: tags (keyword), source (keyword), ingested_at (datetime range)
```

Flows: **text→image** (text tower → search), **image→image** (vision tower → search), both with optional payload filters. Ingestion endpoint runs as a FastAPI `BackgroundTask` with batched embedding (batch 32) and sha256-based idempotency. Corpus for the demo: COCO val + Flickr30k + your own photo set (~40k images ≈ 185 MB of vectors pre-quantization — trivial).

### 6.2 Evaluation

- **Recall@1/5/10** and **mAP@10** on COCO 5k text→image retrieval (standard protocol, comparable to published SigLIP numbers — cite them in the README and show you land within tolerance; reproducing a published number correctly is itself a credibility signal).
- Latency profile: p50/p95 for embed-only, search-only, end-to-end, at 1/8/32 concurrent requests (use `locust`, commit the report).
- Ablation table: full-precision vs int8 quantization recall; 1152-d vs Matryoshka-truncated dims if you evaluate jina-v4 as the documented alternative.

### 6.3 API

`POST /v1/search` (text or image query, `top_k≤50`, `filters`), `POST /v1/search/index` (ingest batch, returns `job_id`), `GET /v1/search/index/{job_id}` (status), `DELETE /v1/search/images/{image_id}`.

---

## 7. Module 4 — OCR + Document Intelligence

### 7.1 Pipeline

```
PDF/image → rasterize (pypdfium2, 200 dpi) → PaddleOCR-VL
  → markdown + per-block text + bounding boxes + layout classes (table/figure/text)
  → GLiNER zero-shot NER over text blocks with request-supplied schema
  → (optional, flag `llm_repair=true`) Qwen3-VL structured-JSON pass for fields
     GLiNER missed, output validated against a Pydantic model, retried once on
     validation failure, else field returned as null with a `warnings` entry
```

PaddleOCR-VL choice rationale: 0.9B params, 109 languages **including Arabic** (your Phase-7 differentiator becomes nearly free), state-of-the-art layout-aware document parsing at its size, Apache 2.0, official vLLM-compatible serving path. DeepSeek-OCR is the documented alternative when the metric is pages/second/GPU (MoE, 570M active params, ~30–40% cheaper per page at scale) — put that comparison in ADR-005. EasyOCR: removed from the design; it's outclassed for document intelligence in 2026.

Isolation rule (repeated because people break it): PaddlePaddle and PyTorch do not share a container. The OCR service has its own image, own lockfile, own CUDA base.

### 7.2 Data & evaluation

- **SROIE** (receipts: company, date, address, total) — field-level extraction F1.
- **FUNSD** (forms) — entity F1 on semantic entities.
- Text metrics: CER/WER via `jiwer` on SROIE transcriptions.
- Arabic track (stretch): evaluate on **KHATT** (handwritten) or synthetic printed Arabic receipts; report CER and honestly document the gap — a measured weakness with analysis beats an unmeasured claim.

### 7.3 API

`POST /v1/ocr` → file (image or PDF ≤20 pages), `schema: [field...]`, `llm_repair: bool`. Response: `{request_id, pages:[{markdown, blocks:[{text, bbox, type}]}], entities:{field: {value, confidence, source_bbox} | null}, warnings[]}`. 202 + job polling for PDFs >3 pages.

---

## 8. Backend — Gateway Design

- **App factory + lifespan**: clients (httpx pools, Qdrant client) created at startup, closed at shutdown; no globals; everything injected via `Depends`.
- **Pydantic v2 everywhere**: request models with `Field` constraints; response models with `model_config = ConfigDict(extra="forbid")`; shared schemas live in `mmp_common` so services and gateway can't drift.
- **Middleware chain** (order matters): request-ID (accept inbound `X-Request-ID` or mint UUIDv7) → structlog context binding → timing/Prometheus → auth (API key header for write/ingest endpoints; demo read endpoints open but rate-limited) → rate limit (`slowapi`, per-key token bucket).
- **Error contract**: RFC 9457 problem+json, one exception handler, never a raw stack trace to the client; stack traces go to structured logs with the request ID.
- **Resilience**: per-service timeouts, 2 retries with jittered backoff on idempotent GETs only, circuit breaker (simple half-open counter) around vLLM so an OOM'd model degrades to fast 503s instead of piled-up 30 s timeouts.
- **OpenAPI**: FastAPI autogenerates; CI publishes the spec as an artifact and fails if endpoints lack response models or descriptions (checked with a small spec-lint script).

---

## 9. Serving & Performance

**vLLM service** (Compose snippet):

```yaml
vllm:
  image: vllm/vllm-openai:latest    # pin exact tag in the repo
  command: >
    --model Qwen/Qwen3-VL-4B-Instruct
    --max-model-len 8192
    --gpu-memory-utilization 0.85
    --limit-mm-per-prompt image=2
    --enable-lora
    --lora-modules caption-lora=/adapters/caption vqa-lora=/adapters/vqa
    --max-lora-rank 16
  volumes: ["./artifacts/adapters:/adapters:ro", "hf_cache:/root/.cache/huggingface"]
  deploy: {resources: {reservations: {devices: [{driver: nvidia, count: 1, capabilities: [gpu]}]}}}
  healthcheck: {test: ["CMD", "curl", "-f", "http://localhost:8000/health"], interval: 10s, retries: 30}
```

Serving targets on a 12 GB card (state these in the README and then *measure* them): caption p95 < 4 s, VQA p95 < 3 s, search end-to-end p95 < 300 ms, OCR < 8 s/page. Optimizations in order of ROI: bf16 → prefix caching (system prompt reuse) → int8 KV-cache if memory-bound → AWQ 4-bit serving build as a documented stretch. If LoRA-on-multimodal hits a wall in your pinned vLLM version, the fallback is two merged checkpoints served by config switch — the gateway contract doesn't change (this is why the OpenAI-compatible boundary matters).

Embedding service: plain FastAPI + Transformers with dynamic batching (asyncio queue, flush at 32 items or 20 ms). Torch inference mode, bf16, warmup at `/readyz`.

---

## 10. Frontend

**Streamlit app (weeks, not months):** four tabs mirroring the API — Caption (upload → caption, latency badge, model version), VQA (upload + question, history), Search (text box or image upload → thumbnail grid with scores and filters), Documents (upload → rendered markdown, entity table with confidence, bbox overlay on the page image). Talks only to the gateway with an API key from env. One `api_client.py` wrapping httpx with the same error taxonomy.

**React stretch (Phase 7):** Vite + React 18 + TypeScript + Tailwind + TanStack Query; same endpoints; deploy static build behind the gateway's nginx sidecar. Nothing in the platform depends on it.

---

## 11. MLOps

### 11.1 CI/CD (GitHub Actions)

```
ci.yml (every PR):
  lint:  uv sync --frozen → ruff check → ruff format --check
  types: mypy --strict packages/
  test:  pytest tests/unit -q --cov=packages --cov-fail-under=80
  build: docker buildx bake (gateway, embeddings, ocr) with GH cache; smoke-start
         gateway container, curl /healthz
release.yml (on tag v*):
  push images to GHCR with git-sha + semver tags
  deploy job (SSH or HF Spaces sync) — gated on manual approval
```

GPU-dependent integration tests don't run in CI (no GPU runners on free tier); they run locally via `make test-integration` against the `dev` compose profile and are required by the release checklist — documented honestly in `docs/runbook.md` rather than pretended away.

### 11.2 Experiment tracking & model registry

W&B: one project per module; every run stores config, git SHA, dataset manifest hash, metrics, sample tables. Best adapters promoted by tagging the W&B artifact `production` and pushing to HF Hub; the serving compose pulls adapters by pinned revision. Model cards (data, metrics, license, limitations, intended use) are mandatory before promotion — enforce it on yourself with a `make promote` script that refuses without a card.

### 11.3 Monitoring

- Prometheus scrapes gateway + services: `http_request_duration_seconds` histogram (per route/status), `inference_requests_total`, `gpu_memory_used_bytes` (via DCGM exporter or a lightweight `pynvml` gauge), Qdrant's built-in metrics.
- Grafana: one "Platform" dashboard (RPS, p50/p95 per endpoint, error rate, GPU memory) — commit the JSON.
- **Drift check (simple, honest):** nightly job embeds the day's search queries, computes mean cosine distance to a reference centroid, alerts (log + Grafana annotation) past a threshold. It's basic, and the README should call it a starting point — that framing shows judgment.
- Logs: structlog JSON to stdout; request ID joins gateway and service logs.

### 11.4 Deployment target

Primary: a single GPU VM (e.g. spot L4/RTX 4000 Ada class) running the full compose behind Caddy (auto-TLS). Public demo fallback: HF Spaces (ZeroGPU) running a trimmed Streamlit + merged-adapter setup — keep both paths in the repo; the Space is your always-on portfolio link, the VM is the "real deployment" story.

---

## 12. Testing Strategy

| Layer | Scope | Runs in CI |
|---|---|---|
| Unit | schemas, answer-normalization util, image preprocessing (incl. the left/right-flip rule), prompt templating, VQA metric against 10 hand-checked fixtures, gateway routing with mocked clients (`respx`) | yes |
| Contract | recorded vLLM/OpenAI responses replayed against the client wrapper; Qdrant against `qdrant` in a service container | yes |
| Integration | real services via compose `dev` profile, tiny fixture set (20 images), asserts end-to-end shapes and status codes | local, pre-release |
| E2E smoke | `make smoke`: bring up `full`, hit all four endpoints with fixtures, assert < thresholds | local, pre-release |
| Quality gates | ruff, mypy strict, coverage ≥ 80% on `packages/`, spec-lint | yes |

Model-quality regression: each promoted adapter must beat the previous on the frozen eval slice (500 held-out examples, exact seed), checked by `make eval-gate`.

---

## 13. Documentation Plan

- Root `README.md`: 90-second pitch, architecture diagram, demo GIF, live demo link, results tables (baseline vs zero-shot vs fine-tuned; retrieval recall; OCR F1), quickstart (`docker compose --profile full up`).
- `docs/adr/`: 001 shared-VLM+LoRA, 002 SigLIP-2 vs jina-v4 (license), 003 Qdrant vs FAISS, 004 PaddleOCR-VL vs DeepSeek-OCR, 005 vLLM serving + fallback, 006 monorepo/uv.
- Per-module `README` with reproduce-my-numbers commands.
- `docs/runbook.md`: start/stop, the one-GPU train-vs-serve rule, adapter promotion, backup/restore of Qdrant snapshots.
- Model cards on HF Hub for every promoted adapter.
- Technical blog post (Phase 7): "Serving two multimodal tasks from one model with runtime LoRA" — the most differentiated artifact this project produces.

---

## 14. Roadmap (mapped to your 6-month plan)

| Phase | Weeks | Deliverable (definition of done) |
|---|---|---|
| 0 Setup | 1 | Repo scaffold, uv workspace, CI green on a hello-world gateway, W&B project, data manifest + download script verified |
| 1 Foundations | 2–5 | Shared preprocessing package w/ tests; COCO/VQAv2 EDA notebook; baseline dataloaders benchmarked (imgs/s) |
| 2 Captioning | 6–9 | Baseline trained + evaluated; caption QLoRA adapter promoted; 3-row metrics table; `/v1/caption` live behind gateway |
| 3 VQA | 10–13 | vqa-lora promoted; official-metric eval + per-type breakdown + gray-image ablation; `/v1/vqa` live |
| 4 Search | 14–17 | Qdrant collection + ingestion; recall/mAP report vs published SigLIP-2 numbers; locust latency report; `/v1/search` live |
| 5 OCR + integration | 18–22 | OCR service + GLiNER + repair pass; SROIE/FUNSD F1 report; all four endpoints on one compose; OpenAPI published |
| 6 MLOps + deploy | 23–26 | Full compose w/ monitoring; CI/CD incl. release; VM deployment + HF Space; Streamlit UI; README with results and GIF |
| 7 Stretch | 27+ | AWQ serving build; Arabic OCR eval; agentic router (small LLM routes free-form requests to modules); React UI; blog post |

Scope-cut order if time slips (decide now, not in month 5): React UI → agentic router → Arabic track → DeepSeek-OCR comparison → drift job. The four modules + deployment + monitoring are the non-negotiable core.

---

## 15. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| vLLM multimodal-LoRA regression in a new release | Medium | Pin the image tag; merged-checkpoint fallback (§9); gateway contract unchanged |
| 12 GB VRAM insufficient for a training config | Medium | 4B default, NF4, grad ckpt, seq 1024; documented Colab/Kaggle resume path |
| VQA metric implementation error inflating results | Medium | Fixture tests against official examples; sanity-check vs published zero-shot numbers |
| PaddlePaddle/PyTorch dependency clash | High if colocated | Hard rule: separate container (§7.1) |
| Dataset licensing questions in a public portfolio | Low | manifest.yaml records license per dataset; README states research/educational use |
| Single-GPU contention (train vs serve) | Certain | `make train` guard + runbook rule |
| Scope creep (the real killer) | High | Phase gates with definition-of-done; pre-committed cut order (§14) |

---

## 16. What Makes This Read as "Production" (checklist for the final review)

1. One published port, internal services isolated, health-gated startup.
2. Typed contracts shared between gateway and services; `extra="forbid"`.
3. Every model choice has an ADR with a rejected alternative and a reason.
4. Metrics reproduce published baselines within tolerance before you claim improvements.
5. Latency numbers are measured (locust reports committed), not asserted.
6. Failure paths are designed: circuit breaker, problem+json, degraded modes.
7. Promotion of a model requires a card and an eval gate.
8. The README's first screen answers: what it does, how it's built, proof it works, how to run it.
