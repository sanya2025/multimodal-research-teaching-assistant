"""mrta.multimodal.vlm_client — VLM captioning via Ollama."""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

import ollama

from mrta.core.exceptions import LLMError

if TYPE_CHECKING:
    from PIL import Image


class VLMClient:
    """Ollama-based vision-language model client.

    Reads settings.ollama_vlm_model by default (currently qwen2.5vl:7b).
    Converts a PIL image to PNG bytes and calls ollama.chat with images=[base64].
    """

    _DEFAULT_PROMPT = "Explain this figure for a graduate student. Be concrete."

    def __init__(self, model: str | None = None) -> None:
        from mrta.core.config import settings

        self._model = model or settings.ollama_vlm_model

    @classmethod
    def is_available(cls, model: str | None = None) -> bool:
        """Return True if the configured VLM model is installed in Ollama."""
        from mrta.core.config import settings

        target = model or settings.ollama_vlm_model
        try:
            ollama.show(target)
            return True
        except Exception:
            return False

    def caption(self, image: Image.Image, prompt: str | None = None) -> str:
        """Caption a PIL image using the configured VLM. Returns the explanation."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        try:
            resp = ollama.chat(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt or self._DEFAULT_PROMPT,
                        "images": [img_b64],
                    }
                ],
                options={"temperature": 0.2},
            )
        except ollama.ResponseError as e:
            if e.status_code == 404 or "not found" in str(e).lower():
                raise LLMError(
                    f"Vision model '{self._model}' is not installed. "
                    f"Run: ollama pull {self._model}"
                ) from e
            raise LLMError(f"Ollama VLM call failed (model={self._model}): {e}") from e
        except Exception as e:
            raise LLMError(f"Ollama VLM call failed (model={self._model}): {e}") from e
        return resp["message"]["content"]
