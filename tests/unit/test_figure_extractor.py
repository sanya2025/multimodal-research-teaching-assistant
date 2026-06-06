"""Tests for mrta.ingestion.figure_extractor."""

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")

from mrta.core.schemas import FigureRecord  # noqa: E402
from mrta.ingestion.figure_extractor import extract_figures  # noqa: E402

FIXTURE_PDF = Path(__file__).parent.parent / "fixtures" / "sample.pdf"


def test_extract_figures_returns_list():
    result = extract_figures(FIXTURE_PDF)
    assert isinstance(result, list)


def test_extract_figures_items_are_figure_records():
    result = extract_figures(FIXTURE_PDF)
    for item in result:
        assert isinstance(item, FigureRecord)


def test_figure_record_image_bytes_non_empty():
    result = extract_figures(FIXTURE_PDF)
    for fig in result:
        assert len(fig.image_bytes) > 0


def test_figure_record_page_and_index_positive():
    result = extract_figures(FIXTURE_PDF)
    for fig in result:
        assert fig.page >= 1
        assert fig.figure_index >= 1


def test_to_pil_returns_pil_image():
    from PIL import Image

    result = extract_figures(FIXTURE_PDF)
    for fig in result:
        img = fig.to_pil()
        assert isinstance(img, Image.Image)
