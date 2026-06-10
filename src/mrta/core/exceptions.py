"""Domain exceptions for the mrta library."""

from __future__ import annotations


class MRTAError(Exception):
    """Base exception for all mrta library errors."""


class IngestionError(MRTAError):
    """Raised when PDF loading, chunking, or figure extraction fails."""


class EmbeddingError(MRTAError):
    """Raised when text or image embedding fails."""


class RetrievalError(MRTAError):
    """Raised when FAISS index operations fail."""


class LLMError(MRTAError):
    """Raised when an LLM or VLM call fails."""


class EvaluationError(MRTAError):
    """Raised when evaluation pipeline or metric computation fails."""
