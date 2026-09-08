"""mrta.retrieval.clip_embedder — CLIP text/image embeddings via HuggingFace Transformers.

Pinned to ``openai/clip-vit-base-patch32``. Both encoders project into the same
512-dimensional space, so a text query embedding and a figure image embedding can
be compared directly by inner product (== cosine, since all vectors are L2-normalized).

Why this exists alongside ``mrta.multimodal.clip_embedder``
-----------------------------------------------------------
``mrta.multimodal.clip_embedder.CLIPEmbedder`` loads OpenAI CLIP weights through
open_clip as ``ViT-B-32`` + ``pretrained="openai"``. That combination silently
mismatches the activation function: OpenAI CLIP was trained with QuickGELU, but
open_clip's plain ``ViT-B-32`` config uses standard GELU. open_clip warns about it
and proceeds. The resulting embeddings differ measurably from true OpenAI CLIP
(cosine ~0.96 image / ~0.98 text) and can reorder retrieval results.

This module loads the model through HuggingFace Transformers, where the correct
QuickGELU activation is part of the published config — verified to agree with
open_clip's ``ViT-B-32-quickgelu`` at cosine 1.000000.

CLIP vectors live in their own embedding space and must never be pooled with
scores from MRTA's text embedder (nomic-embed-text). Cross-stream combination
belongs to rank fusion, not raw score comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from PIL import Image

# The single supported model. Not configurable by design: the evaluation
# provenance records this exact identifier, and swapping it silently would
# invalidate comparisons against previously measured results.
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_EMBEDDING_DIM = 512


class CLIPEmbedder:
    """Text and image encoder for ``openai/clip-vit-base-patch32``.

    Both :meth:`embed_text` and :meth:`embed_image` return 1-D ``float32`` arrays
    of shape ``(512,)`` with unit L2 norm (within 1e-5), so ``a @ b`` is cosine
    similarity and FAISS ``IndexFlatIP`` is a cosine index.

    The model is loaded lazily on first use, so constructing an instance is cheap
    and unit tests can inject a stub without downloading weights.
    """

    def __init__(self, device: str | None = None) -> None:
        """
        Args:
            device: Torch device string ("cpu", "cuda", "mps"). Defaults to "cpu"
                for reproducibility — CLIP inference on 3-figure corpora is fast
                enough on CPU, and CPU avoids nondeterminism across GPU backends.
        """
        self._device = device or "cpu"
        self._model: Any = None
        self._processor: Any = None
        self._warmed_up = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """The exact HuggingFace model identifier used for embeddings."""
        return CLIP_MODEL_ID

    @property
    def dim(self) -> int:
        """Shared embedding dimension for both text and image projections."""
        return CLIP_EMBEDDING_DIM

    @property
    def device(self) -> str:
        """Torch device the model runs on."""
        return self._device

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> tuple[Any, Any]:
        """Load model + processor on first use. Model is put in eval mode."""
        if self._model is None:
            from transformers import CLIPModel, CLIPProcessor

            model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
            model = model.to(self._device)
            model.eval()  # disable dropout — inference must be deterministic
            self._model = model
            self._processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        return self._model, self._processor

    def warmup(self) -> None:
        """Load the model and initialize torch's threading runtime. Idempotent.

        This must run before FAISS is imported or used in the same process.
        faiss-cpu and torch both link against libomp; if FAISS initializes the
        OpenMP runtime first, torch's forward pass segfaults (SIGSEGV) on macOS.
        Executing one forward pass here establishes torch's runtime first, after
        which both libraries coexist safely.

        ImageStore calls this before touching FAISS, so callers normally do not
        need to invoke it directly. Setting OMP_NUM_THREADS=1 also avoids the
        crash, but that serializes all OpenMP work and relies on the environment
        being configured correctly at launch.
        """
        if self._warmed_up:
            return
        # A real forward pass is required — merely constructing the model does
        # not initialize the threading runtime that causes the conflict.
        self.embed_text("warmup")
        self._warmed_up = True

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_text(self, query: str) -> np.ndarray:
        """Embed a text query. Returns a float32 L2-normalized (512,) vector."""
        import torch

        model, processor = self._ensure_model()
        inputs = processor(text=[query], return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.inference_mode():
            features = model.get_text_features(**inputs)
        return self._to_unit_vector(self._extract_embedding(features))

    def embed_image(self, image_input: str | Path | Image.Image) -> np.ndarray:
        """Embed an image. Returns a float32 L2-normalized (512,) vector.

        Args:
            image_input: A PIL Image, or a path to an image file. Images are
                converted to RGB — CLIP's preprocessor expects 3 channels, and
                figure PNGs are frequently greyscale or RGBA.

        Raises:
            FileNotFoundError: the given path does not exist.
            ValueError: the file exists but is not a readable image.
        """
        import torch

        model, processor = self._ensure_model()
        image = self._load_rgb_image(image_input)
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.inference_mode():
            features = model.get_image_features(**inputs)
        return self._to_unit_vector(self._extract_embedding(features))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_rgb_image(image_input: str | Path | Image.Image) -> Image.Image:
        """Resolve a PIL image or a filesystem path to an RGB PIL image."""
        from PIL import Image as PILImage
        from PIL import UnidentifiedImageError

        if isinstance(image_input, PILImage.Image):
            return image_input.convert("RGB")

        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image path does not exist: {path}")
        try:
            with PILImage.open(path) as img:
                return img.convert("RGB")
        except UnidentifiedImageError as e:
            raise ValueError(f"Not a readable image file: {path}") from e

    @staticmethod
    def _extract_embedding(features: Any) -> np.ndarray:
        """Pull the projected embedding out of a transformers feature output.

        transformers >= 5 returns BaseModelOutputWithPooling from
        get_text_features/get_image_features, where pooler_output holds the
        projected vector. Earlier versions return the tensor directly.
        """
        tensor = getattr(features, "pooler_output", features)
        return tensor[0].detach().cpu().numpy().astype("float32")

    @staticmethod
    def _to_unit_vector(vec: np.ndarray) -> np.ndarray:
        """L2-normalize to unit length.

        A zero vector cannot be normalized; returning it unchanged keeps the
        contract of "finite float32, shape (512,)" rather than emitting NaNs
        that would silently poison a FAISS index.
        """
        vec = np.asarray(vec, dtype="float32").reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            return vec
        return (vec / norm).astype("float32")
