"""Typed configuration loaded from .env. Single source of truth for app settings."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider ---
    llm_provider: Literal["ollama", "huggingface", "openai", "anthropic", "google"] = "ollama"

    ollama_host: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.2:3b"
    ollama_vlm_model: str = "llava:7b"

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


settings = Settings()
