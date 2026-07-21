"""mrta.core.llm — provider-agnostic LLM client (Ollama-only for now)."""

from __future__ import annotations

import ollama

from mrta.core.exceptions import LLMError


class LLMClient:
    """Thin wrapper around an LLM provider.

    Reads settings.llm_provider and settings.ollama_llm_model by default.
    """

    def __init__(self, provider: str | None = None, model: str | None = None) -> None:
        from mrta.core.config import settings

        self._provider = provider or settings.llm_provider
        self._model = model or settings.ollama_llm_model

    @property
    def model(self) -> str:
        """The resolved model name."""
        return self._model

    def chat(self, messages: list[dict], temperature: float = 0.1) -> str:
        """Send a chat request and return the response text.

        Args:
            messages: OpenAI-style list of role/content dicts.
            temperature: Sampling temperature (0.0–1.0).
        """
        try:
            resp = ollama.chat(
                model=self._model,
                messages=messages,
                options={"temperature": temperature},
            )
        except Exception as e:
            raise LLMError(f"Ollama chat failed (model={self._model}): {e}") from e
        self._last_usage = {
            "prompt_tokens": resp.get("prompt_eval_count", 0),
            "response_tokens": resp.get("eval_count", 0),
        }
        return resp["message"]["content"]
