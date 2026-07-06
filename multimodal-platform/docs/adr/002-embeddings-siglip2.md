# ADR-002: SigLIP-2 (so400m/384) for retrieval embeddings

**Status:** accepted

**Decision:** SigLIP-2 as the dual encoder for text→image and image→image search.
Apache 2.0, strongest open image-text similarity model as of mid-2026, native
Transformers support since v4.49.

**Alternative rejected:** jina-embeddings-v4 — better on visually rich documents and
multilingual queries, but CC-BY-NC (non-commercial). Kept as an evaluated comparison
in the retrieval report, not in the production path.

**Constraint accepted:** 64-token text limit; gateway truncates and surfaces a warning.
