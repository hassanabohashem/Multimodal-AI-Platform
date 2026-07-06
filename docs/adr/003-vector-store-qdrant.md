# ADR-003: Qdrant over FAISS for the serving index

**Status:** accepted

**Decision:** Qdrant (Docker) with cosine distance, HNSW (m=16, ef_construct=128),
int8 scalar quantization from day one, payload indexes on tags/source.

**Alternative rejected:** FAISS — excellent library, but it is an index, not a
service: no payload filtering, persistence, snapshots, or metrics out of the box.
FAISS remains the in-notebook tool for evaluation sweeps.
