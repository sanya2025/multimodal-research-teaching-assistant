"""Unit tests for mrta.retrieval.clip_embedder.

Split into two groups:

* Pure-logic tests exercise normalization, embedding extraction, and image input
  handling with no model weights at all — these always run.
* Contract tests load the real openai/clip-vit-base-patch32 and verify the numeric
  guarantees callers depend on (shape, dtype, unit norm, determinism). They are
  opt-in, because they need a ~600 MB download that CI should not depend on::

      MRTA_CLIP_MODEL_TESTS=1 pytest tests/unit/test_retrieval_clip_embedder.py

  They must also run in a process where faiss has not been imported. faiss-cpu
  and torch both link libomp, and a torch forward pass after faiss initializes
  OpenMP segfaults on macOS (see CLIPEmbedder.warmup). Production code avoids
  this by warming torch first; a pytest session that already collected a
  faiss-using module cannot, so the fixture skips rather than crashing the run.

Note this module tests mrta.retrieval.clip_embedder (HuggingFace Transformers).
mrta.multimodal.clip_embedder is a separate open_clip-based implementation with
its own tests in tests/unit/test_clip_embedder.py.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from mrta.retrieval.clip_embedder import (
    CLIP_EMBEDDING_DIM,
    CLIP_MODEL_ID,
    CLIPEmbedder,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_clip() -> CLIPEmbedder:
    """A CLIPEmbedder with real weights, loaded at most once for this module."""
    if os.getenv("MRTA_CLIP_MODEL_TESTS") != "1":
        pytest.skip("set MRTA_CLIP_MODEL_TESTS=1 to run tests against real CLIP weights")
    if "faiss" in sys.modules:
        pytest.skip(
            "faiss is already imported in this process; a torch forward pass would "
            "segfault (see CLIPEmbedder.warmup). Run this file on its own."
        )
    pytest.importorskip("transformers", reason="transformers not installed")
    pytest.importorskip("torch", reason="torch not installed")
    embedder = CLIPEmbedder()
    try:
        embedder.embed_text("probe")  # forces the weights to load
    except Exception as e:  # offline CI, no cache, download blocked
        pytest.skip(f"CLIP weights unavailable: {type(e).__name__}: {e}")
    return embedder


@pytest.fixture
def rgb_image() -> Image.Image:
    return Image.new("RGB", (32, 32), color=(120, 90, 200))


@pytest.fixture
def png_path(tmp_path, rgb_image):
    p = tmp_path / "figure.png"
    rgb_image.save(p)
    return p


# ---------------------------------------------------------------------------
# Model identity — no weights needed
# ---------------------------------------------------------------------------


class TestModelIdentity:
    def test_model_id_is_pinned(self) -> None:
        assert CLIP_MODEL_ID == "openai/clip-vit-base-patch32"

    def test_embedder_reports_pinned_model(self) -> None:
        assert CLIPEmbedder().model_name == "openai/clip-vit-base-patch32"

    def test_dim_is_512(self) -> None:
        assert CLIP_EMBEDDING_DIM == 512
        assert CLIPEmbedder().dim == 512

    def test_default_device_is_cpu(self) -> None:
        assert CLIPEmbedder().device == "cpu"

    def test_device_override_respected(self) -> None:
        assert CLIPEmbedder(device="meta").device == "meta"

    def test_construction_does_not_load_weights(self) -> None:
        """Constructing must stay cheap so tests and CLI startup are fast."""
        assert CLIPEmbedder()._model is None


# ---------------------------------------------------------------------------
# Normalization — pure function, no weights
# ---------------------------------------------------------------------------


class TestUnitNormalization:
    def test_normalizes_to_unit_length(self) -> None:
        out = CLIPEmbedder._to_unit_vector(np.array([3.0, 4.0], dtype="float32"))
        assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-6)

    def test_preserves_direction(self) -> None:
        out = CLIPEmbedder._to_unit_vector(np.array([3.0, 4.0], dtype="float32"))
        np.testing.assert_allclose(out, [0.6, 0.8], atol=1e-6)

    def test_output_is_float32(self) -> None:
        out = CLIPEmbedder._to_unit_vector(np.array([1.0, 2.0], dtype="float64"))
        assert out.dtype == np.float32

    def test_flattens_to_1d(self) -> None:
        out = CLIPEmbedder._to_unit_vector(np.array([[3.0, 4.0]], dtype="float32"))
        assert out.shape == (2,)

    def test_zero_vector_returned_unchanged_not_nan(self) -> None:
        """A zero vector cannot be normalized; NaNs would poison the FAISS index."""
        out = CLIPEmbedder._to_unit_vector(np.zeros(4, dtype="float32"))
        assert not np.isnan(out).any()
        np.testing.assert_array_equal(out, np.zeros(4, dtype="float32"))

    def test_already_normalized_is_stable(self) -> None:
        v = np.array([1.0, 0.0, 0.0], dtype="float32")
        np.testing.assert_allclose(CLIPEmbedder._to_unit_vector(v), v, atol=1e-6)


# ---------------------------------------------------------------------------
# Embedding extraction — transformers 4.x vs 5.x output shapes
# ---------------------------------------------------------------------------


class FakeTensor:
    """Minimal stand-in for a torch tensor: supports [0].detach().cpu().numpy().

    Using a stub rather than real torch keeps these tests independent of the
    faiss/torch OpenMP conflict described in the module docstring, and makes the
    contract of _extract_embedding explicit — it only needs a tensor-like object.
    """

    def __init__(self, rows: list[list[float]], dtype: str = "float32") -> None:
        self._array = np.asarray(rows, dtype=dtype)

    def __getitem__(self, idx: int) -> FakeTensor:
        return FakeTensor(self._array[idx].tolist(), dtype=str(self._array.dtype))

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._array


class TestEmbeddingExtraction:
    def test_extracts_pooler_output_when_present(self) -> None:
        """transformers >= 5 returns BaseModelOutputWithPooling."""
        features = SimpleNamespace(pooler_output=FakeTensor([[1.0, 2.0, 3.0]]))
        np.testing.assert_allclose(
            CLIPEmbedder._extract_embedding(features), [1.0, 2.0, 3.0], atol=1e-6
        )

    def test_falls_back_to_raw_tensor(self) -> None:
        """transformers < 5 returns the tensor directly."""
        out = CLIPEmbedder._extract_embedding(FakeTensor([[4.0, 5.0]]))
        np.testing.assert_allclose(out, [4.0, 5.0], atol=1e-6)

    def test_output_is_float32(self) -> None:
        out = CLIPEmbedder._extract_embedding(FakeTensor([[1.0, 2.0]], dtype="float64"))
        assert out.dtype == np.float32

    def test_pooler_output_preferred_over_object_itself(self) -> None:
        """When both are viable, pooler_output wins — that is the projected vector."""
        features = SimpleNamespace(pooler_output=FakeTensor([[7.0, 8.0]]))
        np.testing.assert_allclose(CLIPEmbedder._extract_embedding(features), [7.0, 8.0], atol=1e-6)


# ---------------------------------------------------------------------------
# Image input handling — no weights needed
# ---------------------------------------------------------------------------


class TestImageInputHandling:
    def test_accepts_pil_image(self, rgb_image) -> None:
        assert CLIPEmbedder._load_rgb_image(rgb_image).mode == "RGB"

    def test_accepts_path_object(self, png_path) -> None:
        assert CLIPEmbedder._load_rgb_image(png_path).mode == "RGB"

    def test_accepts_string_path(self, png_path) -> None:
        assert CLIPEmbedder._load_rgb_image(str(png_path)).mode == "RGB"

    def test_converts_greyscale_to_rgb(self, tmp_path) -> None:
        """Figure PNGs are often greyscale; CLIP's processor needs 3 channels."""
        p = tmp_path / "grey.png"
        Image.new("L", (16, 16), color=128).save(p)
        assert CLIPEmbedder._load_rgb_image(p).mode == "RGB"

    def test_converts_rgba_to_rgb(self, tmp_path) -> None:
        p = tmp_path / "alpha.png"
        Image.new("RGBA", (16, 16), color=(1, 2, 3, 128)).save(p)
        assert CLIPEmbedder._load_rgb_image(p).mode == "RGB"

    def test_missing_path_raises_file_not_found(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            CLIPEmbedder._load_rgb_image(tmp_path / "absent.png")

    def test_non_image_file_raises_value_error(self, tmp_path) -> None:
        p = tmp_path / "notanimage.png"
        p.write_text("this is plain text, not a PNG", encoding="utf-8")
        with pytest.raises(ValueError, match="Not a readable image"):
            CLIPEmbedder._load_rgb_image(p)


# ---------------------------------------------------------------------------
# Numeric contract — requires real weights
# ---------------------------------------------------------------------------


class TestRealModelContract:
    def test_text_embedding_shape(self, real_clip) -> None:
        assert real_clip.embed_text("attention mechanism").shape == (512,)

    def test_text_embedding_dtype(self, real_clip) -> None:
        assert real_clip.embed_text("attention mechanism").dtype == np.float32

    def test_text_embedding_unit_norm(self, real_clip) -> None:
        v = real_clip.embed_text("attention mechanism")
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5

    def test_image_embedding_shape(self, real_clip, rgb_image) -> None:
        assert real_clip.embed_image(rgb_image).shape == (512,)

    def test_image_embedding_dtype(self, real_clip, rgb_image) -> None:
        assert real_clip.embed_image(rgb_image).dtype == np.float32

    def test_image_embedding_unit_norm(self, real_clip, rgb_image) -> None:
        v = real_clip.embed_image(rgb_image)
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5

    def test_image_embedding_from_path_matches_pil(self, real_clip, rgb_image, png_path) -> None:
        """Path and PIL inputs must produce the same vector for the same image."""
        from_pil = real_clip.embed_image(rgb_image)
        from_path = real_clip.embed_image(png_path)
        assert float(from_pil @ from_path) == pytest.approx(1.0, abs=1e-5)

    def test_text_embedding_deterministic(self, real_clip) -> None:
        a = real_clip.embed_text("scaled dot-product attention")
        b = real_clip.embed_text("scaled dot-product attention")
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_image_embedding_deterministic(self, real_clip, rgb_image) -> None:
        a = real_clip.embed_image(rgb_image)
        b = real_clip.embed_image(rgb_image)
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_different_text_gives_different_vectors(self, real_clip) -> None:
        a = real_clip.embed_text("a transformer architecture diagram")
        b = real_clip.embed_text("a photograph of a cat")
        assert float(a @ b) < 0.99

    def test_text_and_image_are_comparable(self, real_clip, rgb_image) -> None:
        """Shared space: the dot product must be a finite cosine in [-1, 1]."""
        sim = float(real_clip.embed_text("a purple square") @ real_clip.embed_image(rgb_image))
        assert -1.0 <= sim <= 1.0

    def test_warmup_is_idempotent(self, real_clip) -> None:
        real_clip.warmup()
        real_clip.warmup()
        assert real_clip._warmed_up is True
