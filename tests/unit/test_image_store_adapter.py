"""Unit tests for mrta.retrieval.image_store_adapter.

The adapter is what lets the legacy MultimodalRetriever read the CLIP index that
production ingestion actually writes. These tests pin the two properties that
bug depended on: the adapter returns EvidenceRecords (not VisualRecords), and the
evidence_id it assigns matches the caption index's id for the same figure, so RRF
fuses them as one piece of evidence rather than two.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mrta.core.schemas import EvidenceRecord, VisualRecord
from mrta.retrieval.image_store_adapter import ImageStoreAdapter, visual_record_to_evidence

DOC = "attention_is_all_you_need_a639448e61"


def make_visual_record(
    page: int = 3,
    figure_index: int = 1,
    document_id: str = DOC,
    image_path: str = "data/figures/x_p3_f1.png",
) -> VisualRecord:
    return VisualRecord(
        document_id=document_id,
        page=page,
        figure_id=f"fig_p{page}_{figure_index}",
        figure_index=figure_index,
        image_path=image_path,
        extraction_method="raster_crop",
    )


def make_image_store(hits: list[tuple[VisualRecord, float]]) -> MagicMock:
    store = MagicMock()
    store.search.return_value = hits
    store.size = len(hits)
    return store


class TestVisualRecordToEvidence:
    def test_returns_evidence_record(self) -> None:
        ev = visual_record_to_evidence(make_visual_record())
        assert isinstance(ev, EvidenceRecord)

    def test_modality_is_image(self) -> None:
        """The legacy path splits evidence on modality; 'image' is what routes it to visual."""
        assert visual_record_to_evidence(make_visual_record()).modality == "image"

    def test_evidence_id_matches_caption_index_form(self) -> None:
        """RRF dedups on evidence_id, so this must equal the caption record's id."""
        ev = visual_record_to_evidence(make_visual_record(page=3, figure_index=1))
        assert ev.evidence_id == f"{DOC}_p3_f1"

    def test_image_path_is_carried(self) -> None:
        ev = visual_record_to_evidence(make_visual_record(image_path="data/figures/a.png"))
        assert ev.image_path == "data/figures/a.png"

    def test_source_used_when_resolved(self) -> None:
        ev = visual_record_to_evidence(make_visual_record(), source="attention.pdf")
        assert ev.source == "attention.pdf"

    def test_source_falls_back_to_document_id(self) -> None:
        """Degraded but truthful: never invent a filename."""
        assert visual_record_to_evidence(make_visual_record()).source == DOC

    def test_page_and_figure_index_survive(self) -> None:
        ev = visual_record_to_evidence(make_visual_record(page=4, figure_index=2))
        assert (ev.page, ev.figure_index) == (4, 2)


class TestImageStoreAdapterSearch:
    def test_search_with_scores_returns_evidence_and_score(self) -> None:
        rec = make_visual_record()
        adapter = ImageStoreAdapter(make_image_store([(rec, 0.83)]))
        results = adapter.search_with_scores("transformer architecture", k=5)
        assert len(results) == 1
        ev, score = results[0]
        assert isinstance(ev, EvidenceRecord)
        assert score == pytest.approx(0.83)

    def test_search_returns_records_only(self) -> None:
        adapter = ImageStoreAdapter(make_image_store([(make_visual_record(), 0.7)]))
        assert all(isinstance(r, EvidenceRecord) for r in adapter.search("q"))

    def test_k_is_forwarded_as_top_k(self) -> None:
        """MultimodalRetriever calls k=; ImageStore expects top_k=."""
        store = make_image_store([])
        ImageStoreAdapter(store).search_with_scores("q", k=3)
        store.search.assert_called_once_with("q", top_k=3)

    def test_empty_index_returns_empty_list(self) -> None:
        assert ImageStoreAdapter(make_image_store([])).search_with_scores("q") == []

    def test_size_reflects_underlying_store(self) -> None:
        adapter = ImageStoreAdapter(make_image_store([(make_visual_record(), 0.5)]))
        assert adapter.size == 1

    def test_resolver_supplies_source(self) -> None:
        adapter = ImageStoreAdapter(
            make_image_store([(make_visual_record(), 0.9)]),
            source_resolver=lambda doc_id: "attention.pdf" if doc_id == DOC else None,
        )
        ev, _ = adapter.search_with_scores("q")[0]
        assert ev.source == "attention.pdf"

    def test_resolver_failure_does_not_break_retrieval(self) -> None:
        """Source resolution is cosmetic; a raising resolver must not lose the figure."""

        def boom(doc_id: str) -> str:
            raise RuntimeError("index unavailable")

        adapter = ImageStoreAdapter(
            make_image_store([(make_visual_record(), 0.9)]), source_resolver=boom
        )
        results = adapter.search_with_scores("q")
        assert len(results) == 1
        assert results[0][0].source == DOC

    def test_adapter_exposes_no_write_surface(self) -> None:
        """Indexing stays with ImageStore so exactly one component writes the CLIP index."""
        adapter = ImageStoreAdapter(make_image_store([]))
        assert not hasattr(adapter, "add")
        assert not hasattr(adapter, "add_images")
