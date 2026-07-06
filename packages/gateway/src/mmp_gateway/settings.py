"""Environment-driven gateway configuration."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All gateway config; every field overridable via MMP_* env vars."""

    model_config = SettingsConfigDict(env_prefix="MMP_", env_file=".env", extra="ignore")

    api_key: str = "change-me-demo-key"
    internal_token: str = "change-me-internal-token"
    rate_limit: str = "30/minute"

    vllm_url: str = "http://vllm:8000/v1"
    embeddings_url: str = "http://embeddings:8002"
    ocr_url: str = "http://ocr:8003"
    qdrant_url: str = "http://qdrant:6333"

    caption_model: str = "caption-lora"
    vqa_model: str = "vqa-lora"
    base_model: str = "Qwen/Qwen3-VL-4B-Instruct"

    timeout_caption_s: float = 30
    timeout_vqa_s: float = 30
    timeout_search_s: float = 5
    timeout_ocr_s: float = 120
    max_image_mb: int = 10


settings = Settings()
