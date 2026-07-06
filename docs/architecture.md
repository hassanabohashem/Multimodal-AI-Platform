# Architecture

See the full design document (docs/design.md) for rationale. Summary:

- The gateway owns zero model weights; it can restart instantly and be tested without a GPU.
- All inference services expose /healthz, /readyz, /metrics; compose gates startup on health.
- Contracts live in `mmp_common.schemas` and use `extra="forbid"` — the gateway and services cannot drift.
- Failure design: per-service timeouts, circuit breaker around vLLM, RFC 9457 problem+json, Retry-After on 503.
- One-GPU rule: training and serving never share the card (`make train-guard`).
