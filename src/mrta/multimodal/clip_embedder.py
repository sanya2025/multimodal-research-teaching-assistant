"""mrta.multimodal.clip_embedder — CLIP image and text embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL import Image


class CLIPEmbedder:
    """CLIP ViT-B-32 wrapper for image and text embedding in a shared space.

    Requires the [multimodal] optional dependency group (open-clip-torch).
    Both embed_image and embed_text return float32 L2-normalised (512,) vectors,
    so dot-product is cosine similarity — directly comparable across modalities.
    """

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai") -> None:
        import open_clip  # lazy: optional [multimodal] dep
        import torch

        self._torch = torch
        self._model_name = f"{model_name}/{pretrained}"
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._model = model.eval()
        self._preprocess = preprocess
        self._dim: int = model.visual.output_dim

    @property
    def model_name(self) -> str:
        """Model identifier string, e.g. 'ViT-B-32/openai'."""
        return self._model_name

    @property
    def dim(self) -> int:
        """Shared embedding dimension for image and text projections."""
        return self._dim

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Embed a PIL image. Returns float32 L2-normalised (512,) vector."""
        with self._torch.no_grad():
            tensor = self._preprocess(image.convert("RGB")).unsqueeze(0)
            feat = self._model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat[0].cpu().numpy().astype("float32")

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a text string. Returns float32 L2-normalised (512,) vector."""
        with self._torch.no_grad():
            tokens = self._tokenizer([text])
            feat = self._model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat[0].cpu().numpy().astype("float32")
