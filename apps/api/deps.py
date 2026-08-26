"""Shared FastAPI dependency functions — read from app.state set by lifespan."""

from __future__ import annotations

from fastapi import Request


def get_store(request: Request):
    """Return the VectorStore attached to app.state."""
    return request.app.state.store


def get_llm(request: Request):
    """Return the LLMClient attached to app.state."""
    return request.app.state.llm


def get_retriever(request: Request):
    """Return the MultimodalRetriever, or None if the multimodal extra is not installed."""
    return getattr(request.app.state, "retriever", None)


def get_vlm(request: Request):
    """Return the VLMClient used for multimodal generation, or None if unavailable."""
    return getattr(request.app.state, "vlm", None)
