"""Unit tests for CaptionVectorStore.

All tests mock Embedder and use pytest.importorskip("faiss") so they skip
cleanly if the [retrieval] extra is not installed.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from mrta.core.schemas import EvidenceRecord
from mrta.eval.adapter import EvalAdapter
from mrta.eval.types import CanonicalEvidence
from mrta.retrieval.caption_store import CaptionVectorStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 4


def make_embedder(dim: int = DIM) -> MagicMock:
    embedder = MagicMock()
    embedder.dim = dim
    embedder.model_name = "mock-embed"
    return embedder


def make_record(
    evidence_id: str,
    caption: str | None = None,
    nearby_text: str | None = None,
    modality: str = "image",
    page: int = 1,
    figure_index: int | None = 1,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        doc_id="doc1",
        source="test.pdf",
        page=page,
        modality=modality,
        caption=caption,
        nearby_text=nearby_text,
        figure_index=figure_index,
    )


def unit_vec(dim: int, nonzero: int) -> np.ndarray:
    """Return a (1, dim) float32 array with a 1.0 at position `nonzero`."""
    v = np.zeros((1, dim), dtype="float32")
    v[0, nonzero] = 1.0
    return v


# ---------------------------------------------------------------------------
# TestCaptionVectorStoreAdd
# ---------------------------------------------------------------------------


class TestCaptionVectorStoreAdd:
    def test_empty_store_size_zero(self) -> None:
        store = CaptionVectorStore(make_embedder())
        assert store.size == 0

    def test_add_single_record_increases_size(self) -> None:
        pytest.importorskip("faiss")
        embedder = make_embedder()
        embedder.embed.return_value = unit_vec(DIM, 0)
        store = CaptionVectorStore(embedder)
        store.add([make_record("r1", caption="a diagram")])
        assert store.size == 1

    def test_add_multiple_records(self) -> None:
        pytest.importorskip("faiss")
        embedder = make_embedder()
        embedder.embed.return_value = np.ones((2, DIM), dtype="float32") / DIM**0.5
        store = CaptionVectorStore(embedder)
        store.add([make_record("r1"), make_record("r2")])
        assert store.size == 2

    def test_add_empty_list_is_noop(self) -> None:
        embedder = make_embedder()
        store = CaptionVectorStore(embedder)
        store.add([])
        assert store.size == 0
        embedder.embed.assert_not_called()

    def test_add_embeds_retrieval_text(self) -> None:
        pytest.importorskip("faiss")
        embedder = make_embedder()
        embedder.embed.return_value = unit_vec(DIM, 0)
        store = CaptionVectorStore(embedder)
        rec = make_record("r1", caption="attention mechanism")
        store.add([rec])
        texts_embedded = embedder.embed.call_args[0][0]
        assert texts_embedded == ["attention mechanism"]

    def test_add_uses_nearby_text_when_no_caption(self) -> None:
        pytest.importorskip("faiss")
        embedder = make_embedder()
        embedder.embed.return_value = unit_vec(DIM, 0)
        store = CaptionVectorStore(embedder)
        rec = make_record("r1", nearby_text="residual connections")
        store.add([rec])
        texts_embedded = embedder.embed.call_args[0][0]
        assert texts_embedded == ["residual connections"]


# ---------------------------------------------------------------------------
# TestCaptionVectorStoreSearch
# ---------------------------------------------------------------------------


class TestCaptionVectorStoreSearch:
    def _two_record_store(self) -> tuple[CaptionVectorStore, EvidenceRecord, EvidenceRecord]:
        """Store with two orthogonal records: r_arch (dim 0) and r_chart (dim 1)."""
        pytest.importorskip("faiss")
        embedder = make_embedder()
        store = CaptionVectorStore(embedder)
        rec_a = make_record("r_arch", caption="encoder decoder architecture", page=3)
        rec_b = make_record("r_chart", caption="BLEU score comparison chart", page=7)
        add_embs = np.vstack([unit_vec(DIM, 0), unit_vec(DIM, 1)])
        embedder.embed.return_value = add_embs
        store.add([rec_a, rec_b])
        return store, rec_a, rec_b

    def test_search_empty_store_returns_empty(self) -> None:
        embedder = make_embedder()
        embedder.embed.return_value = unit_vec(DIM, 0)
        store = CaptionVectorStore(embedder)
        assert store.search("diagram") == []

    def test_search_returns_evidence_records(self) -> None:
        store, _, _ = self._two_record_store()
        store._embedder.embed.return_value = unit_vec(DIM, 0)
        results = store.search("architecture", k=1)
        assert len(results) == 1
        assert isinstance(results[0], EvidenceRecord)

    def test_search_top1_is_closest_match(self) -> None:
        store, rec_a, _ = self._two_record_store()
        store._embedder.embed.return_value = unit_vec(DIM, 0)
        results = store.search("architecture diagram", k=1)
        assert results[0].evidence_id == "r_arch"

    def test_retrieval_score_set_on_results(self) -> None:
        store, _, _ = self._two_record_store()
        store._embedder.embed.return_value = unit_vec(DIM, 0)
        results = store.search("architecture", k=1)
        assert results[0].retrieval_score is not None
        assert results[0].retrieval_score > 0.0

    def test_search_with_scores_returns_pairs(self) -> None:
        store, _, _ = self._two_record_store()
        store._embedder.embed.return_value = unit_vec(DIM, 0)
        pairs = store.search_with_scores("architecture", k=1)
        assert len(pairs) == 1
        record, score = pairs[0]
        assert isinstance(record, EvidenceRecord)
        assert isinstance(score, float)

    def test_score_in_pair_matches_record_score(self) -> None:
        store, _, _ = self._two_record_store()
        store._embedder.embed.return_value = unit_vec(DIM, 0)
        pairs = store.search_with_scores("architecture", k=1)
        record, score = pairs[0]
        assert record.retrieval_score == pytest.approx(score)

    def test_k_larger_than_store_returns_all(self) -> None:
        store, _, _ = self._two_record_store()
        store._embedder.embed.return_value = np.ones((1, DIM), dtype="float32") / DIM**0.5
        results = store.search("anything", k=100)
        assert len(results) == 2

    def test_modality_metadata_preserved(self) -> None:
        store, _, _ = self._two_record_store()
        store._embedder.embed.return_value = unit_vec(DIM, 0)
        results = store.search("architecture", k=2)
        assert all(r.modality == "image" for r in results)

    def test_page_metadata_preserved(self) -> None:
        store, rec_a, _ = self._two_record_store()
        store._embedder.embed.return_value = unit_vec(DIM, 0)
        results = store.search("architecture", k=1)
        assert results[0].page == rec_a.page

    def test_stored_record_not_mutated_by_search(self) -> None:
        """search() must return copies; the internal record must remain unchanged."""
        pytest.importorskip("faiss")
        embedder = make_embedder()
        store = CaptionVectorStore(embedder)
        rec = make_record("r1", caption="test figure")
        embedder.embed.return_value = unit_vec(DIM, 0)
        store.add([rec])
        embedder.embed.return_value = unit_vec(DIM, 0)
        store.search("test", k=1)
        embedder.embed.return_value = unit_vec(DIM, 0)
        store.search("test", k=1)
        assert store._records[0].retrieval_score is None

    def test_page_modality_records_retrievable(self) -> None:
        pytest.importorskip("faiss")
        embedder = make_embedder()
        store = CaptionVectorStore(embedder)
        rec = make_record(
            "r_page", nearby_text="full page render", modality="page", figure_index=None
        )
        embedder.embed.return_value = unit_vec(DIM, 0)
        store.add([rec])
        embedder.embed.return_value = unit_vec(DIM, 0)
        results = store.search("page render", k=1)
        assert results[0].modality == "page"
        assert results[0].evidence_id == "r_page"


# ---------------------------------------------------------------------------
# TestCaptionVectorStorePersistence
# ---------------------------------------------------------------------------


class TestCaptionVectorStorePersistence:
    def _saved_store(
        self, tmp_path, image_bytes: bytes | None = b"\x89PNG\r\n"
    ) -> tuple[CaptionVectorStore, EvidenceRecord]:
        """Build a one-record store and save it to tmp_path."""
        pytest.importorskip("faiss")
        embedder = make_embedder()
        store = CaptionVectorStore(embedder)
        img_path = str(tmp_path / "fig1.png")
        rec = EvidenceRecord(
            evidence_id="doc1_p3_f1",
            doc_id="doc_attention_2017",
            source="attention_is_all_you_need.pdf",
            page=3,
            modality="image",
            figure_index=1,
            caption="The Transformer architecture with encoder and decoder stacks.",
            image_bytes=image_bytes,
            image_path=img_path,
        )
        embedder.embed.return_value = unit_vec(DIM, 0)
        store.add([rec])
        store.save(tmp_path / "caption_index")
        return store, rec

    def test_image_bytes_not_serialized(self, tmp_path) -> None:
        """image_bytes must be absent from persisted metadata."""
        self._saved_store(tmp_path)
        lines = (tmp_path / "caption_index" / "metadata.jsonl").read_text().splitlines()
        record_dict = json.loads(lines[0])
        assert record_dict.get("image_bytes") is None

    def test_metadata_survives_save_load(self, tmp_path) -> None:
        """doc_id, page, figure_index, caption, image_path survive round-trip."""
        _, original = self._saved_store(tmp_path)
        embedder = make_embedder()
        loaded = CaptionVectorStore.load(tmp_path / "caption_index", embedder)
        r = loaded._records[0]
        assert r.evidence_id == original.evidence_id
        assert r.doc_id == original.doc_id
        assert r.page == original.page
        assert r.figure_index == original.figure_index
        assert r.caption == original.caption
        assert r.image_path == original.image_path

    def test_image_bytes_none_after_load(self, tmp_path) -> None:
        """Loaded records must have image_bytes=None."""
        self._saved_store(tmp_path)
        embedder = make_embedder()
        loaded = CaptionVectorStore.load(tmp_path / "caption_index", embedder)
        assert loaded._records[0].image_bytes is None

    def test_image_path_available_after_load(self, tmp_path) -> None:
        """image_path reference survives round-trip for lazy loading."""
        _, original = self._saved_store(tmp_path)
        embedder = make_embedder()
        loaded = CaptionVectorStore.load(tmp_path / "caption_index", embedder)
        assert loaded._records[0].image_path == original.image_path

    def test_retrieval_unchanged_after_reload(self, tmp_path) -> None:
        """Top-1 result and score are identical before and after save/load."""
        pytest.importorskip("faiss")
        _, _ = self._saved_store(tmp_path)
        embedder = make_embedder()
        loaded = CaptionVectorStore.load(tmp_path / "caption_index", embedder)
        embedder.embed.return_value = unit_vec(DIM, 0)
        pairs = loaded.search_with_scores("transformer architecture", k=1)
        assert len(pairs) == 1
        record, score = pairs[0]
        assert record.evidence_id == "doc1_p3_f1"
        assert score == pytest.approx(1.0)

    def test_index_files_written(self, tmp_path) -> None:
        self._saved_store(tmp_path)
        index_dir = tmp_path / "caption_index"
        assert (index_dir / "index.faiss").exists()
        assert (index_dir / "metadata.jsonl").exists()
        assert (index_dir / "config.json").exists()


# ---------------------------------------------------------------------------
# TestEvalAdapterCaptionRecord
# ---------------------------------------------------------------------------

_MANIFEST = {
    "documents": [
        {
            "document_id": "doc_attention_2017",
            "filename": "attention_is_all_you_need.pdf",
            "figures": [
                {"figure_id": "fig_transformer_arch", "page_number": 3, "figure_index": 1},
                {"figure_id": "fig_attention_mechanisms", "page_number": 4, "figure_index": 1},
                {"figure_id": "fig_attention_mechanisms", "page_number": 4, "figure_index": 2},
            ],
        }
    ]
}


def _caption_record(page: int, figure_index: int, figure_id_hint: str = "") -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=f"doc_attention_2017_p{page}_f{figure_index}",
        doc_id="doc_attention_2017",
        source="attention_is_all_you_need.pdf",
        page=page,
        modality="image",
        figure_index=figure_index,
        caption=f"Figure {figure_index} on page {page}. {figure_id_hint}",
    )


class TestEvalAdapterCaptionRecord:
    def test_from_caption_record_resolves_figure_id(self) -> None:
        adapter = EvalAdapter(_MANIFEST)
        rec = _caption_record(3, 1)
        cand = adapter.from_caption_record(rec, score=0.9, rank=1)
        assert cand.evidence.figure_id == "fig_transformer_arch"
        assert cand.evidence.page_number == 3
        assert cand.evidence.document_id == "doc_attention_2017"

    def test_from_caption_record_matches_correct_target(self) -> None:
        adapter = EvalAdapter(_MANIFEST)
        rec = _caption_record(3, 1)
        cand = adapter.from_caption_record(rec, score=0.9, rank=1)
        target = CanonicalEvidence(
            document_id="doc_attention_2017",
            page_number=3,
            figure_id="fig_transformer_arch",
        )
        assert target.matches(cand.evidence)

    def test_from_caption_record_wrong_figure_no_match(self) -> None:
        """A record on the correct page with the wrong figure_index must not match."""
        adapter = EvalAdapter(_MANIFEST)
        rec_wrong = _caption_record(3, 99)  # figure_index 99 not in manifest
        cand = adapter.from_caption_record(rec_wrong, score=0.8, rank=2)
        target = CanonicalEvidence(
            document_id="doc_attention_2017",
            page_number=3,
            figure_id="fig_transformer_arch",
        )
        assert not target.matches(cand.evidence)

    def test_from_caption_record_score_and_rank_preserved(self) -> None:
        adapter = EvalAdapter(_MANIFEST)
        rec = _caption_record(4, 1)
        cand = adapter.from_caption_record(rec, score=0.75, rank=3)
        assert cand.score == pytest.approx(0.75)
        assert cand.rank == 3
