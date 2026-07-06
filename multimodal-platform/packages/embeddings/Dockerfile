FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock* ./
COPY packages/common packages/common
COPY packages/embeddings packages/embeddings
RUN uv sync --frozen --package mmp-embeddings --no-dev || uv sync --package mmp-embeddings --no-dev
EXPOSE 8002
CMD ["uv", "run", "uvicorn", "mmp_embeddings.main:app", "--host", "0.0.0.0", "--port", "8002"]
