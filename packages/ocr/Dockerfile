FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y python3.12 python3.12-venv curl && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml ./
COPY src src
RUN uv venv && uv pip install -e .
EXPOSE 8003
CMD ["uv", "run", "uvicorn", "mmp_ocr.main:app", "--host", "0.0.0.0", "--port", "8003"]
