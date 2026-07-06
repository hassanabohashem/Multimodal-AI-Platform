# ADR-006: uv workspace monorepo

**Status:** accepted

**Decision:** Single repo, uv workspace with packages per service, one lockfile for
the PyTorch side; the OCR package is deliberately excluded (ADR-004). Shared contracts
live in mmp-common so schemas cannot drift between gateway and services.

**Alternative rejected:** polyrepo — coordination overhead for a solo project;
Poetry — slower, weaker workspace story in 2026.
