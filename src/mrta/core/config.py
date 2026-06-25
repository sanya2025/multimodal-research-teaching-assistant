"""Typed configuration. Priority: env vars > .env > configs/{MRTA_ENV}.yaml > defaults."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

# Repo root is three levels above this file: src/mrta/core/config.py
_REPO_ROOT = Path(__file__).parents[3]


class _YamlConfigSource(PydanticBaseSettingsSource):
    """Loads configs/{MRTA_ENV}.yaml as the lowest-priority settings source."""

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        env_name = os.getenv("MRTA_ENV", "dev")
        yaml_path = _REPO_ROOT / "configs" / f"{env_name}.yaml"
        self._data: dict[str, Any] = {}
        if yaml_path.exists():
            with open(yaml_path) as f:
                self._data = yaml.safe_load(f) or {}

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return {k: v for k, v in self._data.items() if v is not None}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider ---
    llm_provider: Literal["ollama", "huggingface", "openai", "anthropic", "google"] = "ollama"

    ollama_host: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.2:latest"
    ollama_vlm_model: str = "qwen2.5vl:latest"

    huggingface_hub_token: str | None = None
    hf_llm_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    hf_vlm_model: str = "Qwen/Qwen2-VL-2B-Instruct"

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None

    # --- Embeddings ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    clip_model: str = "openai/clip-vit-base-patch32"

    # --- Vector store ---
    vector_store: Literal["faiss", "qdrant"] = "faiss"
    qdrant_url: str = "http://localhost:6333"
    vector_store_path: Path = Field(default=Path("data/vector_store"))

    # --- Retrieval ---
    top_k: int = 5
    chunk_size: int = 700
    chunk_overlap: int = 100

    # --- Observability ---
    log_level: str = "INFO"
    log_file: Path = Field(default=Path("data/logs/runs.jsonl"))
    enable_tracing: bool = False
    otel_service_name: str = "mrta"
    otel_exporter_otlp_endpoint: str = ""
    otel_console_exporter: bool = False

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priority: init > env vars > .env > YAML > Python defaults
        return (init_settings, env_settings, dotenv_settings, _YamlConfigSource(settings_cls))


settings = Settings()
