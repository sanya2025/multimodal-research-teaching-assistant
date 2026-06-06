"""Tests for mrta.multimodal.clip_embedder."""

import numpy as np
import pytest

open_clip = pytest.importorskip("open_clip", reason="open-clip-torch not installed")

from PIL import Image  # noqa: E402

from mrta.multimodal.clip_embedder import CLIPEmbedder  # noqa: E402


@pytest.fixture(scope="module")
def clip():
    return CLIPEmbedder()


@pytest.fixture
def white_image():
    return Image.new("RGB", (1, 1), color=(255, 255, 255))


def test_embed_image_shape(clip, white_image):
    emb = clip.embed_image(white_image)
    assert emb.shape == (512,)


def test_embed_image_dtype(clip, white_image):
    emb = clip.embed_image(white_image)
    assert emb.dtype == np.float32


def test_embed_image_l2_norm(clip, white_image):
    emb = clip.embed_image(white_image)
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-5


def test_embed_text_shape_and_norm(clip):
    emb = clip.embed_text("attention mechanism")
    assert emb.shape == (512,)
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-5


def test_image_text_dot_product_positive(clip, white_image):
    img_emb = clip.embed_image(white_image)
    txt_emb = clip.embed_text("white image")
    assert float(img_emb @ txt_emb) > 0.0
