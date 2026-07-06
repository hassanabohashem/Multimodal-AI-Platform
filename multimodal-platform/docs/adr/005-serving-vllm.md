# ADR-005: vLLM OpenAI-compatible serving, with a merged-checkpoint fallback

**Status:** accepted

**Decision:** vLLM serves the VLM with `--enable-lora` and two named adapters. The
gateway speaks only the OpenAI chat API.

**Risk & fallback:** multimodal LoRA support can regress between vLLM releases. Image
tag is pinned in docker-compose.yml. Fallback: merge each adapter into a full
checkpoint (`peft merge_and_unload`) and serve by config switch — zero gateway changes,
which is the point of the OpenAI-compatible boundary.
