"""Validation tests for the v1 benchmark dataset.

Ensures the benchmark file is structurally correct and consistent with the
corpus manifest. These tests prevent accidental corruption across PRs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
QUERIES_PATH = REPO_ROOT / "data" / "eval" / "queries_v1.json"
MANIFEST_PATH = REPO_ROOT / "data" / "eval" / "corpus" / "v1" / "manifest.json"


@pytest.fixture(scope="module")
def dataset() -> dict:
    return json.loads(QUERIES_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def queries(dataset: dict) -> list[dict]:
    return dataset["queries"]


# ---------------------------------------------------------------------------
# Count checks
# ---------------------------------------------------------------------------


def test_total_query_count(queries: list[dict]) -> None:
    assert len(queries) == 20


def test_text_query_count(queries: list[dict]) -> None:
    assert sum(1 for q in queries if q["intent"] == "text") == 8


def test_visual_query_count(queries: list[dict]) -> None:
    assert sum(1 for q in queries if q["intent"] == "visual") == 6


def test_hybrid_query_count(queries: list[dict]) -> None:
    assert sum(1 for q in queries if q["intent"] == "hybrid") == 6


# ---------------------------------------------------------------------------
# Schema checks
# ---------------------------------------------------------------------------


def test_unique_query_ids(queries: list[dict]) -> None:
    ids = [q["query_id"] for q in queries]
    assert len(ids) == len(set(ids))


def test_valid_intent_values(queries: list[dict]) -> None:
    valid = {"text", "visual", "hybrid"}
    for q in queries:
        assert q["intent"] in valid, f"{q['query_id']} has invalid intent {q['intent']!r}"


def test_non_empty_query_text(queries: list[dict]) -> None:
    for q in queries:
        assert q["query"].strip(), f"{q['query_id']} has empty query text"


def test_non_empty_target_evidence(queries: list[dict]) -> None:
    for q in queries:
        assert q["target_evidence"], f"{q['query_id']} has empty target_evidence"


def test_target_evidence_fields(queries: list[dict]) -> None:
    for q in queries:
        for t in q["target_evidence"]:
            assert "document_id" in t, f"{q['query_id']}: missing document_id"
            assert "page_number" in t, f"{q['query_id']}: missing page_number"
            assert "figure_id" in t, f"{q['query_id']}: missing figure_id key"
            assert isinstance(t["page_number"], int), f"{q['query_id']}: page_number not int"


# ---------------------------------------------------------------------------
# Valid document IDs
# ---------------------------------------------------------------------------


def test_valid_document_ids(queries: list[dict], manifest: dict) -> None:
    valid_doc_ids = {doc["document_id"] for doc in manifest["documents"]}
    for q in queries:
        for t in q["target_evidence"]:
            assert t["document_id"] in valid_doc_ids, (
                f"{q['query_id']}: unknown document_id {t['document_id']!r}"
            )


# ---------------------------------------------------------------------------
# Valid figure IDs
# ---------------------------------------------------------------------------


def test_valid_figure_ids(queries: list[dict], manifest: dict) -> None:
    valid_figure_ids: set[str | None] = {None}
    for doc in manifest["documents"]:
        for fig in doc.get("figures", []):
            valid_figure_ids.add(fig["figure_id"])

    for q in queries:
        for t in q["target_evidence"]:
            fid = t.get("figure_id")
            assert fid in valid_figure_ids, (
                f"{q['query_id']}: unknown figure_id {fid!r}"
            )


# ---------------------------------------------------------------------------
# Intent-evidence consistency
# ---------------------------------------------------------------------------


def test_visual_queries_have_figure_target(queries: list[dict]) -> None:
    for q in queries:
        if q["intent"] == "visual":
            has_figure = any(t["figure_id"] is not None for t in q["target_evidence"])
            assert has_figure, f"{q['query_id']}: visual query has no figure target"


def test_text_queries_have_no_figure_target(queries: list[dict]) -> None:
    for q in queries:
        if q["intent"] == "text":
            has_figure = any(t["figure_id"] is not None for t in q["target_evidence"])
            assert not has_figure, f"{q['query_id']}: text query should not have figure target"


def test_hybrid_queries_have_both_text_and_figure(queries: list[dict]) -> None:
    for q in queries:
        if q["intent"] == "hybrid":
            has_figure = any(t["figure_id"] is not None for t in q["target_evidence"])
            has_text = any(t["figure_id"] is None for t in q["target_evidence"])
            assert has_figure, f"{q['query_id']}: hybrid query missing figure target"
            assert has_text, f"{q['query_id']}: hybrid query missing text target"


# ---------------------------------------------------------------------------
# Page numbers within document page count
# ---------------------------------------------------------------------------


def test_page_numbers_within_bounds(queries: list[dict], manifest: dict) -> None:
    doc_pages = {doc["document_id"]: doc["pages"] for doc in manifest["documents"]}
    for q in queries:
        for t in q["target_evidence"]:
            doc_id = t["document_id"]
            page = t["page_number"]
            max_pages = doc_pages.get(doc_id, 0)
            assert 1 <= page <= max_pages, (
                f"{q['query_id']}: page_number {page} out of range [1, {max_pages}] for {doc_id}"
            )


# ---------------------------------------------------------------------------
# Manifest PDF file exists
# ---------------------------------------------------------------------------


def test_corpus_pdf_exists(manifest: dict) -> None:
    corpus_dir = MANIFEST_PATH.parent / "papers"
    for doc in manifest["documents"]:
        pdf_path = corpus_dir / doc["filename"]
        assert pdf_path.exists(), f"Corpus PDF missing: {pdf_path}"


# ---------------------------------------------------------------------------
# Dataset metadata fields
# ---------------------------------------------------------------------------


def test_dataset_version_present(dataset: dict) -> None:
    assert "dataset_version" in dataset
    assert dataset["dataset_version"] == "v1.0.0"


def test_corpus_version_present(dataset: dict) -> None:
    assert "corpus_version" in dataset
    assert dataset["corpus_version"] == "v1.0.0"
