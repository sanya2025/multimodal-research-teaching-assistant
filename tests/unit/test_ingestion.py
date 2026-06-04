"""Unit tests for mrta.ingestion.pdf_loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from mrta.core.schemas import PageRecord, PdfDocument
from mrta.ingestion.pdf_loader import _doc_id, load_pdf

FIXTURE_PDF = Path(__file__).parents[1] / "fixtures" / "sample.pdf"

# Known content written into the fixture PDF (see tests/fixtures/README note)
PAGE_1_PHRASE = "unit testing the mrta ingestion pipeline"
PAGE_2_PHRASE = "Attention mechanisms have become an integral part"


@pytest.fixture(scope="module")
def doc() -> PdfDocument:
    return load_pdf(FIXTURE_PDF)


class TestLoadPdf:
    def test_returns_pdf_document(self, doc: PdfDocument) -> None:
        assert isinstance(doc, PdfDocument)

    def test_page_count(self, doc: PdfDocument) -> None:
        assert doc.n_pages == 2

    def test_pages_list_length(self, doc: PdfDocument) -> None:
        assert len(doc.pages) == 2

    def test_pages_are_page_records(self, doc: PdfDocument) -> None:
        for page in doc.pages:
            assert isinstance(page, PageRecord)

    def test_page_numbers_are_one_indexed(self, doc: PdfDocument) -> None:
        assert doc.pages[0].page == 1
        assert doc.pages[1].page == 2

    def test_source_is_filename(self, doc: PdfDocument) -> None:
        assert doc.source == FIXTURE_PDF.name

    def test_doc_id_matches_pages(self, doc: PdfDocument) -> None:
        for page in doc.pages:
            assert page.doc_id == doc.doc_id

    def test_doc_id_is_stable(self) -> None:
        """Same file loaded twice must produce the same doc_id."""
        doc_a = load_pdf(FIXTURE_PDF)
        doc_b = load_pdf(FIXTURE_PDF)
        assert doc_a.doc_id == doc_b.doc_id

    def test_page_1_text_content(self, doc: PdfDocument) -> None:
        assert PAGE_1_PHRASE in doc.pages[0].text

    def test_page_2_text_content(self, doc: PdfDocument) -> None:
        assert PAGE_2_PHRASE in doc.pages[1].text

    def test_no_images_in_fixture(self, doc: PdfDocument) -> None:
        for page in doc.pages:
            assert page.n_images == 0

    def test_title_from_metadata(self, doc: PdfDocument) -> None:
        assert doc.title == "Sample Fixture Paper"


class TestDocId:
    def test_format(self) -> None:
        """doc_id should be '{stem}_{10-char hex}'."""
        doc_id = _doc_id(FIXTURE_PDF)
        stem, hash_part = doc_id.rsplit("_", 1)
        assert stem == FIXTURE_PDF.stem
        assert len(hash_part) == 10
        assert all(c in "0123456789abcdef" for c in hash_part)

    def test_different_files_produce_different_ids(self, tmp_path: Path) -> None:
        pdf_a = tmp_path / "a.pdf"
        pdf_b = tmp_path / "b.pdf"
        pdf_a.write_bytes(FIXTURE_PDF.read_bytes())
        pdf_b.write_bytes(b"%PDF-1.4 different content")
        assert _doc_id(pdf_a) != _doc_id(pdf_b)
