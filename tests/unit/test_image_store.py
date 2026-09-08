"""Unit tests for mrta.retrieval.image_store.

All tests use a mock CLIPEmbedder returning fixed orthogonal unit vectors, so no
model weights are downloaded and rankings are exactly predictable. faiss is
required; tests skip cleanly without the [retrieval] extra.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from mrta.core.exceptions import RetrievalError
from mrta.core.schemas import VisualRecord
from mrta.eval.adapter import EvalAdapter
from mrta.eval.types import CanonicalEvidence
from mrta.retrieval.image_store import ImageStore

DIM = 512
DOC = "doc_attention_2017"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def unit_vec(nonzero: int, dim: int = DIM) -> np.ndarray:
    """A (dim,) unit vector with 1.0 at position `nonzero`."""
    v = np.zeros(dim, dtype="float32")
    v[nonzero] = 1.0
    return v


def make_embedder(dim: int = DIM) -> MagicMock:
    embedder = MagicMock()
    embedder.dim = dim
    embedder.model_name = "openai/clip-vit-base-patch32"
    embedder.warmup = MagicMock(return_value=None)
    return embedder


def make_record(
    page: int = 3,
    figure_id: str = "fig_transformer_arch",
    figure_index: int = 1,
    image_path: str = "data/figures/f.png",
    document_id: str = DOC,
) -> VisualRecord:
    return VisualRecord(
        document_id=document_id,
        page=page,
        figure_id=figure_id,
        figure_index=figure_index,
        image_path=image_path,
    )


ARCH = make_record(3, "fig_transformer_arch", 1, "data/figures/p3_f1.png")
ATTN_L = make_record(4, "fig_attention_mechanisms", 1, "data/figures/p4_f1.png")
ATTN_R = make_record(4, "fig_attention_mechanisms", 2, "data/figures/p4_f2.png")


def three_record_store() -> tuple[ImageStore, MagicMock]:
    """Store with 3 records on orthogonal axes 0, 1, 2 (insertion order preserved)."""
    pytest.importorskip("faiss")
    embedder = make_embedder()
    embedder.embed_image.side_effect = [unit_vec(0), unit_vec(1), unit_vec(2)]
    store = ImageStore(embedder)
    store.add_images([ARCH, ATTN_L, ATTN_R])
    return store, embedder


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


class TestImageStoreAdd:
    def test_empty_store_size_zero(self) -> None:
        assert ImageStore(make_embedder()).size == 0

    def test_empty_store_add_nothing_is_noop(self) -> None:
        embedder = make_embedder()
        store = ImageStore(embedder)
        store.add_images([])
        assert store.size == 0
        embedder.embed_image.assert_not_called()

    def test_add_single_record(self) -> None:
        pytest.importorskip("faiss")
        embedder = make_embedder()
        embedder.embed_image.return_value = unit_vec(0)
        store = ImageStore(embedder)
        store.add_images([ARCH])
        assert store.size == 1

    def test_add_multiple_records(self) -> None:
        store, _ = three_record_store()
        assert store.size == 3

    def test_embeds_from_image_path_not_bytes(self) -> None:
        """CLIP must embed the image file, never caption or description text."""
        pytest.importorskip("faiss")
        embedder = make_embedder()
        embedder.embed_image.return_value = unit_vec(0)
        ImageStore(embedder).add_images([ARCH])
        embedder.embed_image.assert_called_once_with(ARCH.image_path)

    def test_index_uses_embedder_dim(self) -> None:
        store, _ = three_record_store()
        assert store._ensure_index().d == DIM

    def test_insertion_order_preserved(self) -> None:
        """Deterministic ordering keeps rebuilt indexes comparable."""
        store, _ = three_record_store()
        assert [r.record_id for r in store.records] == [
            ARCH.record_id,
            ATTN_L.record_id,
            ATTN_R.record_id,
        ]

    def test_records_property_returns_copy(self) -> None:
        store, _ = three_record_store()
        store.records.clear()
        assert store.size == 3


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestImageStoreSearch:
    def test_search_empty_store_returns_empty(self) -> None:
        embedder = make_embedder()
        embedder.embed_text.return_value = unit_vec(0)
        assert ImageStore(embedder).search("anything") == []

    def test_top1_is_closest_match(self) -> None:
        store, embedder = three_record_store()
        embedder.embed_text.return_value = unit_vec(1)  # aligns with ATTN_L
        results = store.search("attention", top_k=1)
        assert results[0][0].record_id == ATTN_L.record_id

    def test_results_are_score_ordered(self) -> None:
        store, embedder = three_record_store()
        # Weighted so axis 2 > axis 1 > axis 0
        q = np.zeros(DIM, dtype="float32")
        q[0], q[1], q[2] = 0.1, 0.5, 0.8
        q /= np.linalg.norm(q)
        embedder.embed_text.return_value = q
        scores = [s for _, s in store.search("q", top_k=3)]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_ordering_matches_expected_records(self) -> None:
        store, embedder = three_record_store()
        q = np.zeros(DIM, dtype="float32")
        q[0], q[1], q[2] = 0.1, 0.5, 0.8
        q /= np.linalg.norm(q)
        embedder.embed_text.return_value = q
        ids = [r.record_id for r, _ in store.search("q", top_k=3)]
        assert ids == [ATTN_R.record_id, ATTN_L.record_id, ARCH.record_id]

    def test_k_larger_than_store_returns_all(self) -> None:
        store, embedder = three_record_store()
        embedder.embed_text.return_value = unit_vec(0)
        assert len(store.search("q", top_k=100)) == 3

    def test_k_smaller_than_store_truncates(self) -> None:
        store, embedder = three_record_store()
        embedder.embed_text.return_value = unit_vec(0)
        assert len(store.search("q", top_k=2)) == 2

    def test_retrieval_score_set_on_results(self) -> None:
        store, embedder = three_record_store()
        embedder.embed_text.return_value = unit_vec(0)
        record, score = store.search("q", top_k=1)[0]
        assert record.retrieval_score == pytest.approx(score)

    def test_stored_records_not_mutated_by_search(self) -> None:
        store, embedder = three_record_store()
        embedder.embed_text.return_value = unit_vec(0)
        store.search("q", top_k=3)
        assert all(r.retrieval_score is None for r in store._records)

    def test_canonical_metadata_preserved_through_search(self) -> None:
        store, embedder = three_record_store()
        embedder.embed_text.return_value = unit_vec(2)
        record, _ = store.search("q", top_k=1)[0]
        assert record.document_id == ATTN_R.document_id
        assert record.page == ATTN_R.page
        assert record.figure_id == ATTN_R.figure_id
        assert record.figure_index == ATTN_R.figure_index
        assert record.image_path == ATTN_R.image_path


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestImageStorePersistence:
    def test_save_writes_expected_files(self, tmp_path) -> None:
        store, _ = three_record_store()
        store.save(tmp_path / "idx")
        for name in ("index.faiss", "metadata.jsonl", "config.json"):
            assert (tmp_path / "idx" / name).exists()

    def test_config_records_model_provenance(self, tmp_path) -> None:
        store, _ = three_record_store()
        store.save(tmp_path / "idx")
        cfg = json.loads((tmp_path / "idx" / "config.json").read_text())
        assert cfg["model"] == "openai/clip-vit-base-patch32"
        assert cfg["embedding_dimension"] == DIM
        assert cfg["normalization"] == "L2"
        assert cfg["similarity"] == "inner_product/cosine"
        assert cfg["image_bytes_persisted"] is False

    def test_no_image_bytes_in_metadata(self, tmp_path) -> None:
        """The visual index must stay lightweight — paths only, never pixels."""
        store, _ = three_record_store()
        store.save(tmp_path / "idx")
        raw = (tmp_path / "idx" / "metadata.jsonl").read_text()
        assert "image_bytes" not in raw

    def test_round_trip_preserves_record_count(self, tmp_path) -> None:
        store, _ = three_record_store()
        store.save(tmp_path / "idx")
        assert ImageStore.load(tmp_path / "idx", make_embedder()).size == 3

    def test_round_trip_preserves_canonical_metadata(self, tmp_path) -> None:
        store, _ = three_record_store()
        store.save(tmp_path / "idx")
        loaded = ImageStore.load(tmp_path / "idx", make_embedder())
        for original, restored in zip(store.records, loaded.records, strict=True):
            assert restored.document_id == original.document_id
            assert restored.page == original.page
            assert restored.figure_id == original.figure_id
            assert restored.figure_index == original.figure_index
            assert restored.image_path == original.image_path

    def test_round_trip_preserves_figure_id(self, tmp_path) -> None:
        store, _ = three_record_store()
        store.save(tmp_path / "idx")
        loaded = ImageStore.load(tmp_path / "idx", make_embedder())
        assert [r.figure_id for r in loaded.records] == [
            "fig_transformer_arch",
            "fig_attention_mechanisms",
            "fig_attention_mechanisms",
        ]

    def test_round_trip_preserves_image_path(self, tmp_path) -> None:
        store, _ = three_record_store()
        store.save(tmp_path / "idx")
        loaded = ImageStore.load(tmp_path / "idx", make_embedder())
        assert [r.image_path for r in loaded.records] == [
            ARCH.image_path,
            ATTN_L.image_path,
            ATTN_R.image_path,
        ]

    def test_round_trip_preserves_order(self, tmp_path) -> None:
        store, _ = three_record_store()
        store.save(tmp_path / "idx")
        loaded = ImageStore.load(tmp_path / "idx", make_embedder())
        assert [r.record_id for r in loaded.records] == [r.record_id for r in store.records]

    def test_rankings_identical_after_reload(self, tmp_path) -> None:
        store, embedder = three_record_store()
        q = np.zeros(DIM, dtype="float32")
        q[0], q[1], q[2] = 0.1, 0.5, 0.8
        q /= np.linalg.norm(q)
        embedder.embed_text.return_value = q
        before = [(r.record_id, s) for r, s in store.search("q", top_k=3)]

        store.save(tmp_path / "idx")
        reloaded_embedder = make_embedder()
        reloaded_embedder.embed_text.return_value = q
        after = [
            (r.record_id, s)
            for r, s in ImageStore.load(tmp_path / "idx", reloaded_embedder).search("q", top_k=3)
        ]

        assert [i for i, _ in before] == [i for i, _ in after]
        for (_, s_before), (_, s_after) in zip(before, after, strict=True):
            assert s_before == pytest.approx(s_after, abs=1e-6)

    def test_load_rejects_mismatched_model(self, tmp_path) -> None:
        """Vectors from a different model are not comparable — fail loudly."""
        store, _ = three_record_store()
        store.save(tmp_path / "idx")
        wrong = make_embedder()
        wrong.model_name = "openai/clip-vit-large-patch14"
        with pytest.raises(RetrievalError, match="not comparable"):
            ImageStore.load(tmp_path / "idx", wrong)

    def test_load_missing_index_raises_retrieval_error(self, tmp_path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(RetrievalError, match="Cannot load FAISS index"):
            ImageStore.load(tmp_path / "empty", make_embedder())


# ---------------------------------------------------------------------------
# EvalAdapter integration
# ---------------------------------------------------------------------------

_MANIFEST = {
    "documents": [
        {
            "document_id": DOC,
            "filename": "attention_is_all_you_need.pdf",
            "figures": [
                {"figure_id": "fig_transformer_arch", "page_number": 3, "figure_index": 1},
                {"figure_id": "fig_attention_mechanisms", "page_number": 4, "figure_index": 1},
                {"figure_id": "fig_attention_mechanisms", "page_number": 4, "figure_index": 2},
            ],
        }
    ]
}


class TestEvalAdapterVisualRecord:
    def test_maps_canonical_identity(self) -> None:
        cand = EvalAdapter(_MANIFEST).from_visual_record(ARCH, score=0.9, rank=1)
        assert cand.evidence.document_id == DOC
        assert cand.evidence.page_number == 3
        assert cand.evidence.figure_id == "fig_transformer_arch"

    def test_preserves_score_and_rank(self) -> None:
        cand = EvalAdapter(_MANIFEST).from_visual_record(ATTN_R, score=0.42, rank=3)
        assert cand.score == pytest.approx(0.42)
        assert cand.rank == 3

    def test_candidate_id_is_record_id(self) -> None:
        cand = EvalAdapter(_MANIFEST).from_visual_record(ARCH, score=0.5, rank=1)
        assert cand.candidate_id == ARCH.record_id

    def test_matches_correct_target(self) -> None:
        cand = EvalAdapter(_MANIFEST).from_visual_record(ARCH, score=0.9, rank=1)
        target = CanonicalEvidence(DOC, 3, "fig_transformer_arch")
        assert target.matches(cand.evidence)

    def test_wrong_figure_does_not_match(self) -> None:
        """Retrieving Figure 2 must not count as a hit for a Figure 1 target."""
        cand = EvalAdapter(_MANIFEST).from_visual_record(ATTN_R, score=0.9, rank=1)
        target = CanonicalEvidence(DOC, 3, "fig_transformer_arch")
        assert not target.matches(cand.evidence)

    def test_both_subfigures_match_shared_figure_id(self) -> None:
        """p4/f1 and p4/f2 are one conceptual figure — either satisfies the target."""
        adapter = EvalAdapter(_MANIFEST)
        target = CanonicalEvidence(DOC, 4, "fig_attention_mechanisms")
        for rec in (ATTN_L, ATTN_R):
            assert target.matches(adapter.from_visual_record(rec, 0.9, 1).evidence)
