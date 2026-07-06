FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock* ./
COPY packages/common packages/common
COPY packages/gateway packages/gateway
RUN uv sync --frozen --package mmp-gateway --no-dev || uv sync --package mmp-gateway --no-dev
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "mmp_gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
