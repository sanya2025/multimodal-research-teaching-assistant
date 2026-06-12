"""Citation coverage tests using the golden QA dataset.

Constructs fake retrieved_sources (list[str] of PDF filenames) matching
expected_documents from golden QA items and verifies citation_coverage scores.

TODO: Replace fake sources with real VectorStore.search() output once PDFs are
indexed. The metric function and assertion logic will stay identical.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mrta.evaluation.metrics import citation_coverage
from tests.evaluation.conftest import load_golden_qa

GOLDEN_QA_PATH = Path(__file__).parent / "datasets" / "golden_qa.yaml"


def test_load_golden_qa_returns_all_items() -> None:
    items = load_golden_qa(GOLDEN_QA_PATH)
    assert len(items) == 40


def test_loader_handles_missing_optional_fields() -> None:
    items = load_golden_qa(GOLDEN_QA_PATH)
    for item in items:
        assert "source" in item
        assert item["source"] == ""  # field absent from YAML items — default ""
        assert isinstance(item["expected_documents"], list)
        assert isinstance(item["expected_keywords"], list)


def test_coverage_clip_question() -> None:
    items = load_golden_qa(GOLDEN_QA_PATH)
    clip_q = next(q for q in items if "CLIP.pdf" in q.get("expected_documents", []))
    fake_retrieved = clip_q["expected_documents"]  # perfect retrieval
    score = citation_coverage(fake_retrieved, clip_q["expected_documents"])
    assert score == pytest.approx(1.0)


def test_coverage_wrong_source() -> None:
    items = load_golden_qa(GOLDEN_QA_PATH)
    clip_q = next(q for q in items if "CLIP.pdf" in q.get("expected_documents", []))
    fake_retrieved = ["BLIP-2.pdf", "SigLIP.pdf"]  # completely wrong sources
    score = citation_coverage(fake_retrieved, clip_q["expected_documents"])
    assert score == pytest.approx(0.0)


def test_coverage_empty_expected() -> None:
    assert citation_coverage(["CLIP.pdf", "SigLIP.pdf"], []) == pytest.approx(1.0)
