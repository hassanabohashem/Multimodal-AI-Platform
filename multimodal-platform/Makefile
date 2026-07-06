.PHONY: dev sync lint type test test-integration up down full train-guard smoke

sync:
	uv sync --all-packages

lint:
	uv run ruff check . && uv run ruff format --check .

type:
	uv run mypy packages/

test:
	uv run pytest tests/unit --cov=packages --cov-fail-under=80

test-integration:
	uv run pytest tests/integration -m integration

up:
	docker compose --profile dev up -d

full:
	docker compose --profile full --profile monitoring up -d

down:
	docker compose --profile full --profile monitoring --profile dev down

# One-GPU rule: refuse to train while vLLM is serving (docs/runbook.md).
train-guard:
	@if nvidia-smi | grep -qi vllm; then \
	  echo "ERROR: vLLM is running on the GPU. 'make down' first."; exit 1; fi

train-caption: train-guard
	uv run python -m mmp_training.captioning.qlora.train --config configs/caption_qlora.yaml

train-vqa: train-guard
	uv run python -m mmp_training.vqa.train --config configs/vqa_qlora.yaml

smoke:
	uv run python tests/integration/smoke.py
