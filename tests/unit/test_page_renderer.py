"""Tests for mrta.ingestion.page_renderer."""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")

from mrta.core.schemas import EvidenceRecord  # noqa: E402
from mrta.ingestion.page_renderer import render_page, render_pages  # noqa: E402

FIXTURE_PDF = Path(__file__).parent.parent / "fixtures" / "sample.pdf"


class TestRenderPage:
    def test_returns_evidence_record(self) -> None:
        rec = render_page(FIXTURE_PDF, page_number=1)
        assert isinstance(rec, EvidenceRecord)

    def test_modality_is_page(self) -> None:
        rec = render_page(FIXTURE_PDF, page_number=1)
        assert rec.modality == "page"

    def test_image_bytes_non_empty(self) -> None:
        rec = render_page(FIXTURE_PDF, page_number=1)
        assert rec.image_bytes is not None
        assert len(rec.image_bytes) > 0

    def test_image_bytes_is_valid_png(self) -> None:
        rec = render_page(FIXTURE_PDF, page_number=1)
        assert rec.image_bytes is not None
        assert rec.image_bytes[:4] == b"\x89PNG"

    def test_page_number_set_correctly(self) -> None:
        rec = render_page(FIXTURE_PDF, page_number=1)
        assert rec.page == 1

    def test_source_set_to_filename(self) -> None:
        rec = render_page(FIXTURE_PDF, page_number=1)
        assert rec.source == FIXTURE_PDF.name

    def test_evidence_id_contains_page_number(self) -> None:
        rec = render_page(FIXTURE_PDF, page_number=1)
        assert "_p1_" in rec.evidence_id
        assert rec.evidence_id.endswith("_page")

    def test_dpi_affects_image_size(self) -> None:
        low = render_page(FIXTURE_PDF, page_number=1, dpi=72)
        high = render_page(FIXTURE_PDF, page_number=1, dpi=150)
        assert len(high.image_bytes) > len(low.image_bytes)

    def test_invalid_page_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            render_page(FIXTURE_PDF, page_number=9999)

    def test_page_zero_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            render_page(FIXTURE_PDF, page_number=0)

    def test_pil_roundtrip(self) -> None:
        import io

        from PIL import Image

        rec = render_page(FIXTURE_PDF, page_number=1)
        img = Image.open(io.BytesIO(rec.image_bytes))
        assert img.mode in ("RGB", "RGBA", "L")
        assert img.width > 0


class TestRenderPages:
    def test_returns_list(self) -> None:
        result = render_pages(FIXTURE_PDF)
        assert isinstance(result, list)

    def test_all_items_are_evidence_records(self) -> None:
        result = render_pages(FIXTURE_PDF)
        for item in result:
            assert isinstance(item, EvidenceRecord)

    def test_page_subset(self) -> None:
        result = render_pages(FIXTURE_PDF, pages=[1])
        assert len(result) == 1
        assert result[0].page == 1

    def test_ordering_by_page(self) -> None:
        result = render_pages(FIXTURE_PDF)
        pages = [r.page for r in result]
        assert pages == sorted(pages)
