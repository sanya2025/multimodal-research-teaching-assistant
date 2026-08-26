"""mrta.multimodal.vlm_client — VLM multimodal generation via Ollama."""

from __future__ import annotations

import base64
import io
from collections.abc import Sequence
from typing import TYPE_CHECKING

import ollama

from mrta.core.exceptions import LLMError

if TYPE_CHECKING:
    from PIL import Image


def _pil_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class VLMClient:
    """Ollama-based vision-language model client.

    Low-level API: ``generate(prompt, images)`` — arbitrary multimodal generation.
    Convenience API: ``caption(image, prompt)`` — single-image description (unchanged).

    Both methods raise ``LLMError`` on network or model-not-found failures.
    """

    _DEFAULT_CAPTION_PROMPT = "Explain this figure for a graduate student. Be concrete."

    def __init__(self, model: str | None = None) -> None:
        from mrta.core.config import settings

        self._model = model or settings.ollama_vlm_model

    @property
    def model(self) -> str:
        return self._model

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

    def generate(
        self,
        prompt: str,
        images: Sequence[Image.Image],
        temperature: float = 0.2,
    ) -> str:
        """General-purpose multimodal generation: prompt + one or more images.

        Supports figure description, visual QA, chart interpretation, diagram
        reasoning, multi-image comparison, and multimodal RAG generation.

        Args:
            prompt: Instruction or question for the VLM.
            images: One or more PIL images to attach.
            temperature: Sampling temperature (lower = more deterministic).

        Returns:
            The VLM response as a plain string.
        """
        img_b64s = [_pil_to_b64(img) for img in images]
        try:
            resp = ollama.chat(
                model=self._model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": img_b64s,
                    }
                ],
                options={"temperature": temperature},
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

    def caption(self, image: Image.Image, prompt: str | None = None) -> str:
        """Caption a single PIL image. Convenience wrapper around generate()."""
        return self.generate(
            prompt=prompt or self._DEFAULT_CAPTION_PROMPT,
            images=[image],
        )
