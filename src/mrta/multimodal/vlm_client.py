"""mrta.multimodal.vlm_client — VLM captioning via Ollama."""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

import ollama

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

    def caption(self, image: Image.Image, prompt: str | None = None) -> str:
        """Caption a PIL image using the configured VLM. Returns the explanation."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

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
        return resp["message"]["content"]
