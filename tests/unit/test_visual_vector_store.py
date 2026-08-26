"""Unit tests for VisualVectorStore.

All tests mock CLIPEmbedder — open_clip is not required to run them.
pytest.importorskip("faiss") guards each test that creates a real FAISS index.
PIL (Pillow) is used to produce minimal valid image bytes for records.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from mrta.core.schemas import EvidenceRecord
from mrta.retrieval.visual_vector_store import VisualVectorStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 4


def make_clip(dim: int = DIM) -> MagicMock:
    clip = MagicMock()
    clip.dim = dim
    clip.model_name = "test/clip"
    return clip


def tiny_png() -> bytes:
    """Return bytes of a 1×1 white PNG — valid image_bytes for EvidenceRecord."""
    PIL_Image = pytest.importorskip("PIL.Image")
    buf = io.BytesIO()
    PIL_Image.new("RGB", (1, 1), color=(255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def make_record(
    evidence_id: str,
    with_image: bool = True,
    page: int = 1,
    figure_index: int | None = 1,
    modality: str = "image",
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        doc_id="doc1",
        source="test.pdf",
        page=page,
        modality=modality,
        figure_index=figure_index,
        image_bytes=tiny_png() if with_image else None,
    )


def unit_vec(dim: int, nonzero: int) -> np.ndarray:
    """Return a (dim,) float32 array with 1.0 at position `nonzero`."""
    v = np.zeros(dim, dtype="float32")
    v[nonzero] = 1.0
    return v


# ---------------------------------------------------------------------------
# TestVisualVectorStoreAdd
# ---------------------------------------------------------------------------


class TestVisualVectorStoreAdd:
    def test_empty_store_size_zero(self) -> None:
        store = VisualVectorStore(make_clip())
        assert store.size == 0

    def test_add_single_record_increases_size(self) -> None:
        pytest.importorskip("faiss")
        clip = make_clip()
        clip.embed_image.return_value = unit_vec(DIM, 0)
        store = VisualVectorStore(clip)
        store.add([make_record("r1")])
        assert store.size == 1

    def test_add_multiple_records(self) -> None:
        pytest.importorskip("faiss")
        clip = make_clip()
        clip.embed_image.return_value = unit_vec(DIM, 0)
        store = VisualVectorStore(clip)
        store.add([make_record("r1"), make_record("r2")])
        assert store.size == 2

    def test_add_empty_list_is_noop(self) -> None:
        clip = make_clip()
        store = VisualVectorStore(clip)
        store.add([])
        assert store.size == 0
        clip.embed_image.assert_not_called()

    def test_add_skips_record_without_image_bytes(self) -> None:
        pytest.importorskip("faiss")
        clip = make_clip()
        clip.embed_image.return_value = unit_vec(DIM, 0)
        store = VisualVectorStore(clip)
        store.add([make_record("r_no_img", with_image=False), make_record("r_img")])
        assert store.size == 1

    def test_all_no_image_bytes_stays_empty(self) -> None:
        clip = make_clip()
        store = VisualVectorStore(clip)
        store.add([make_record("r1", with_image=False), make_record("r2", with_image=False)])
        assert store.size == 0
        clip.embed_image.assert_not_called()

    def test_add_calls_embed_image_not_embed_text(self) -> None:
        pytest.importorskip("faiss")
        clip = make_clip()
        clip.embed_image.return_value = unit_vec(DIM, 0)
        store = VisualVectorStore(clip)
        store.add([make_record("r1")])
        clip.embed_image.assert_called_once()
        clip.embed_text.assert_not_called()


# ---------------------------------------------------------------------------
# TestVisualVectorStoreSearch
# ---------------------------------------------------------------------------


class TestVisualVectorStoreSearch:
    def _two_record_store(self) -> tuple[VisualVectorStore, EvidenceRecord, EvidenceRecord]:
        """Store with two orthogonal records: r_arch (dim 0) and r_chart (dim 1)."""
        pytest.importorskip("faiss")
        clip = make_clip()
        store = VisualVectorStore(clip)
        rec_a = make_record("r_arch", page=3)
        rec_b = make_record("r_chart", page=7)
        clip.embed_image.side_effect = [unit_vec(DIM, 0), unit_vec(DIM, 1)]
        store.add([rec_a, rec_b])
        return store, rec_a, rec_b

    def test_search_empty_store_returns_empty(self) -> None:
        clip = make_clip()
        clip.embed_text.return_value = unit_vec(DIM, 0)
        store = VisualVectorStore(clip)
        assert store.search("diagram") == []

    def test_search_returns_evidence_records(self) -> None:
        store, _, _ = self._two_record_store()
        store._clip.embed_text.return_value = unit_vec(DIM, 0)
        results = store.search("architecture diagram", k=1)
        assert len(results) == 1
        assert isinstance(results[0], EvidenceRecord)

    def test_search_top1_is_closest_match(self) -> None:
        store, rec_a, _ = self._two_record_store()
        store._clip.embed_text.return_value = unit_vec(DIM, 0)
        results = store.search("encoder decoder", k=1)
        assert results[0].evidence_id == "r_arch"

    def test_retrieval_score_set_on_results(self) -> None:
        store, _, _ = self._two_record_store()
        store._clip.embed_text.return_value = unit_vec(DIM, 0)
        results = store.search("diagram", k=1)
        assert results[0].retrieval_score is not None
        assert results[0].retrieval_score > 0.0

    def test_search_with_scores_returns_pairs(self) -> None:
        store, _, _ = self._two_record_store()
        store._clip.embed_text.return_value = unit_vec(DIM, 0)
        pairs = store.search_with_scores("architecture", k=1)
        assert len(pairs) == 1
        record, score = pairs[0]
        assert isinstance(record, EvidenceRecord)
        assert isinstance(score, float)

    def test_score_in_pair_matches_record_score(self) -> None:
        store, _, _ = self._two_record_store()
        store._clip.embed_text.return_value = unit_vec(DIM, 0)
        pairs = store.search_with_scores("architecture", k=1)
        record, score = pairs[0]
        assert record.retrieval_score == pytest.approx(score)

    def test_k_larger_than_store_returns_all(self) -> None:
        store, _, _ = self._two_record_store()
        store._clip.embed_text.return_value = unit_vec(DIM, 0)
        results = store.search("anything", k=100)
        assert len(results) == 2

    def test_stored_record_not_mutated_by_search(self) -> None:
        """search() must return copies; the internal record must remain unchanged."""
        pytest.importorskip("faiss")
        clip = make_clip()
        store = VisualVectorStore(clip)
        rec = make_record("r1")
        clip.embed_image.return_value = unit_vec(DIM, 0)
        store.add([rec])
        clip.embed_text.return_value = unit_vec(DIM, 0)
        store.search("test", k=1)
        store.search("test", k=1)
        assert store._records[0].retrieval_score is None

    def test_modality_metadata_preserved(self) -> None:
        store, _, _ = self._two_record_store()
        store._clip.embed_text.return_value = unit_vec(DIM, 0)
        results = store.search("anything", k=2)
        assert all(r.modality == "image" for r in results)

    def test_page_metadata_preserved(self) -> None:
        store, rec_a, _ = self._two_record_store()
        store._clip.embed_text.return_value = unit_vec(DIM, 0)
        results = store.search("architecture", k=1)
        assert results[0].page == rec_a.page

    def test_page_modality_records_retrievable(self) -> None:
        pytest.importorskip("faiss")
        clip = make_clip()
        store = VisualVectorStore(clip)
        rec = make_record("r_page", modality="page", figure_index=None)
        clip.embed_image.return_value = unit_vec(DIM, 0)
        store.add([rec])
        clip.embed_text.return_value = unit_vec(DIM, 0)
        results = store.search("full page render", k=1)
        assert results[0].modality == "page"
        assert results[0].evidence_id == "r_page"


# ---------------------------------------------------------------------------
# TestVisualVectorStorePersistence
# ---------------------------------------------------------------------------


class TestVisualVectorStorePersistence:
    def test_save_creates_expected_files(self, tmp_path: Path) -> None:
        pytest.importorskip("faiss")
        clip = make_clip()
        clip.embed_image.return_value = unit_vec(DIM, 0)
        store = VisualVectorStore(clip)
        store.add([make_record("r1")])
        store.save(tmp_path / "vs")
        assert (tmp_path / "vs" / "index.faiss").exists()
        assert (tmp_path / "vs" / "metadata.jsonl").exists()
        assert (tmp_path / "vs" / "config.json").exists()

    def test_save_load_roundtrip_preserves_top1(self, tmp_path: Path) -> None:
        pytest.importorskip("faiss")
        clip = make_clip()
        clip.embed_image.side_effect = [unit_vec(DIM, 0), unit_vec(DIM, 1)]
        store = VisualVectorStore(clip)
        store.add([make_record("r_arch", page=3), make_record("r_chart", page=7)])
        store.save(tmp_path / "vs")

        clip2 = make_clip()
        reloaded = VisualVectorStore.load(tmp_path / "vs", clip2)

        clip.embed_text.return_value = unit_vec(DIM, 0)
        clip2.embed_text.return_value = unit_vec(DIM, 0)
        original_top = store.search("architecture", k=1)
        reloaded_top = reloaded.search("architecture", k=1)

        assert len(reloaded_top) == 1
        assert reloaded_top[0].evidence_id == original_top[0].evidence_id
        # image_bytes are not persisted (re-fetch from source PDF if needed)
        assert reloaded_top[0].image_bytes is None
