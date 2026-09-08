"""Validation tests for the v2 benchmark dataset (5 papers, 100 queries).

Mirrors tests/evaluation/test_benchmark_validation.py (v1) but validates the
larger, multi-document v2 corpus and its richer evidence-grounding metadata.
v1's test file is untouched — this is an additive file, not a replacement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
QUERIES_PATH = REPO_ROOT / "data" / "eval" / "queries_v2.json"
MANIFEST_PATH = REPO_ROOT / "data" / "eval" / "corpus" / "v2" / "manifest.json"

VALID_INTENTS = {"text", "visual_caption", "visual_layout", "hybrid", "hard_visual"}
VISUAL_INTENTS = {"visual_caption", "visual_layout", "hard_visual"}


@pytest.fixture(scope="module")
def dataset() -> dict:
    return json.loads(QUERIES_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def queries(dataset: dict) -> list[dict]:
    return dataset["queries"]


@pytest.fixture(scope="module")
def valid_document_ids(manifest: dict) -> set[str]:
    return {doc["document_id"] for doc in manifest["documents"]}


@pytest.fixture(scope="module")
def valid_figure_ids(manifest: dict) -> set[str]:
    ids: set[str] = set()
    for doc in manifest["documents"]:
        for fig in doc.get("figures", []):
            ids.add(fig["figure_id"])
    return ids


# ---------------------------------------------------------------------------
# Document-level checks
# ---------------------------------------------------------------------------


def test_five_documents_exist(manifest: dict) -> None:
    assert len(manifest["documents"]) == 5


def test_document_ids_unique(manifest: dict) -> None:
    ids = [d["document_id"] for d in manifest["documents"]]
    assert len(ids) == len(set(ids))


def test_document_ids_match_expected(manifest: dict) -> None:
    expected = {"attention_is_all_you_need", "clip", "siglip", "blip2", "llava"}
    assert {d["document_id"] for d in manifest["documents"]} == expected


def test_all_pdfs_exist_and_readable(manifest: dict) -> None:
    pytest.importorskip("fitz")
    import fitz

    papers_dir = MANIFEST_PATH.parent / "papers"
    for doc in manifest["documents"]:
        pdf_path = papers_dir / doc["filename"]
        assert pdf_path.exists(), f"Missing PDF: {pdf_path}"
        with fitz.open(pdf_path) as fh:
            assert fh.page_count > 0


def test_page_counts_nonzero_and_match_pdf(manifest: dict) -> None:
    pytest.importorskip("fitz")
    import fitz

    papers_dir = MANIFEST_PATH.parent / "papers"
    for doc in manifest["documents"]:
        assert doc["pages"] > 0
        with fitz.open(papers_dir / doc["filename"]) as fh:
            assert fh.page_count == doc["pages"], f"{doc['document_id']}: page count mismatch"


def test_hashes_computable_and_match(manifest: dict) -> None:
    import hashlib

    papers_dir = MANIFEST_PATH.parent / "papers"
    for doc in manifest["documents"]:
        pdf_path = papers_dir / doc["filename"]
        actual = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        assert actual == doc["sha256"], f"{doc['document_id']}: sha256 mismatch"


# ---------------------------------------------------------------------------
# Query-level checks
# ---------------------------------------------------------------------------


def test_total_query_count(queries: list[dict]) -> None:
    assert len(queries) == 100


def test_unique_query_ids(queries: list[dict]) -> None:
    ids = [q["query_id"] for q in queries]
    assert len(ids) == len(set(ids))


def test_valid_intent_values(queries: list[dict]) -> None:
    for q in queries:
        assert q["intent"] in VALID_INTENTS, f"{q['query_id']}: invalid intent {q['intent']!r}"


def test_non_empty_query_text(queries: list[dict]) -> None:
    for q in queries:
        assert q["query"].strip(), f"{q['query_id']}: empty query text"


def test_non_empty_expected_evidence(queries: list[dict]) -> None:
    for q in queries:
        assert q["expected_evidence"], f"{q['query_id']}: empty expected_evidence"


def test_expected_evidence_fields(queries: list[dict]) -> None:
    for q in queries:
        for ev in q["expected_evidence"]:
            assert "document_id" in ev, f"{q['query_id']}: missing document_id"
            assert "page_number" in ev, f"{q['query_id']}: missing page_number"
            assert "figure_id" in ev, f"{q['query_id']}: missing figure_id key"
            assert isinstance(ev["page_number"], int), f"{q['query_id']}: page_number not int"


def test_query_document_id_resolves(queries: list[dict], valid_document_ids: set[str]) -> None:
    for q in queries:
        assert q["document_id"] in valid_document_ids, f"{q['query_id']}: unknown document_id"


def test_evidence_document_id_resolves(queries: list[dict], valid_document_ids: set[str]) -> None:
    for q in queries:
        for ev in q["expected_evidence"]:
            assert (
                ev["document_id"] in valid_document_ids
            ), f"{q['query_id']}: unknown evidence document_id {ev['document_id']!r}"


def test_evidence_figure_id_resolves(queries: list[dict], valid_figure_ids: set[str]) -> None:
    for q in queries:
        for ev in q["expected_evidence"]:
            fid = ev.get("figure_id")
            if fid is not None:
                assert fid in valid_figure_ids, f"{q['query_id']}: unknown figure_id {fid!r}"


def test_page_numbers_within_bounds(queries: list[dict], manifest: dict) -> None:
    doc_pages = {d["document_id"]: d["pages"] for d in manifest["documents"]}
    for q in queries:
        for ev in q["expected_evidence"]:
            max_pages = doc_pages.get(ev["document_id"], 0)
            assert 1 <= ev["page_number"] <= max_pages, (
                f"{q['query_id']}: page {ev['page_number']} out of range "
                f"[1, {max_pages}] for {ev['document_id']}"
            )


def test_queries_per_document_balanced(queries: list[dict]) -> None:
    """20 queries per paper (5 papers x 20 = 100)."""
    from collections import Counter

    counts = Counter(q["document_id"] for q in queries)
    assert len(counts) == 5
    for doc_id, n in counts.items():
        assert n == 20, f"{doc_id}: expected 20 queries, got {n}"


# ---------------------------------------------------------------------------
# Figure-target checks
# ---------------------------------------------------------------------------


def test_all_figure_targets_have_image_artifact(queries: list[dict], manifest: dict) -> None:
    """Every figure_id referenced by a query resolves to an existing image file."""
    path_by_figure_id: dict[str, list[str]] = {}
    for doc in manifest["documents"]:
        for fig in doc.get("figures", []):
            path_by_figure_id.setdefault(fig["figure_id"], []).append(fig["image_path"])

    for q in queries:
        for ev in q["expected_evidence"]:
            fid = ev.get("figure_id")
            if fid is None:
                continue
            paths = path_by_figure_id.get(fid)
            assert paths, f"{q['query_id']}: figure_id {fid!r} has no manifest entry"
            for p in paths:
                assert (REPO_ROOT / p).exists(), f"{fid}: image artifact missing at {p}"


def test_figure_target_document_and_page_match_manifest(
    queries: list[dict], manifest: dict
) -> None:
    """A query's (document_id, page_number, figure_id) must match a manifest entry."""
    manifest_entries: set[tuple[str, int, str]] = set()
    for doc in manifest["documents"]:
        for fig in doc.get("figures", []):
            manifest_entries.add((doc["document_id"], fig["page_number"], fig["figure_id"]))

    for q in queries:
        for ev in q["expected_evidence"]:
            fid = ev.get("figure_id")
            if fid is None:
                continue
            key = (ev["document_id"], ev["page_number"], fid)
            assert key in manifest_entries, f"{q['query_id']}: {key} not found in manifest"


def test_figure_extraction_type_is_valid(manifest: dict) -> None:
    valid_types = {"raster_crop", "page_render_fallback"}
    for doc in manifest["documents"]:
        for fig in doc.get("figures", []):
            assert fig["extraction_type"] in valid_types


def test_page_render_fallback_uses_figure_index_zero(manifest: dict) -> None:
    """Convention: figure_index=0 marks a whole-page render, not a raster crop."""
    for doc in manifest["documents"]:
        for fig in doc.get("figures", []):
            if fig["extraction_type"] == "page_render_fallback":
                assert fig["figure_index"] == 0


def test_raster_crop_uses_positive_figure_index(manifest: dict) -> None:
    for doc in manifest["documents"]:
        for fig in doc.get("figures", []):
            if fig["extraction_type"] == "raster_crop":
                assert fig["figure_index"] >= 1


# ---------------------------------------------------------------------------
# Text-target checks
# ---------------------------------------------------------------------------


def test_text_only_targets_have_null_figure_id(queries: list[dict]) -> None:
    for q in queries:
        if q["intent"] == "text":
            for ev in q["expected_evidence"]:
                assert ev["figure_id"] is None, f"{q['query_id']}: text query has a figure target"


def test_text_targets_reference_valid_pages(queries: list[dict], manifest: dict) -> None:
    doc_pages = {d["document_id"]: d["pages"] for d in manifest["documents"]}
    for q in queries:
        for ev in q["expected_evidence"]:
            if ev["figure_id"] is None:
                assert 1 <= ev["page_number"] <= doc_pages[ev["document_id"]]


# ---------------------------------------------------------------------------
# Hybrid / visual intent-evidence consistency
# ---------------------------------------------------------------------------


def test_visual_intents_have_figure_target(queries: list[dict]) -> None:
    for q in queries:
        if q["intent"] in VISUAL_INTENTS:
            has_figure = any(ev["figure_id"] is not None for ev in q["expected_evidence"])
            assert has_figure, f"{q['query_id']}: {q['intent']} query has no figure target"


def test_hybrid_queries_have_independent_text_and_figure_targets(queries: list[dict]) -> None:
    """Hybrid targets must resolve independently — not collapsed into one page-level target."""
    for q in queries:
        if q["intent"] == "hybrid":
            figure_targets = [ev for ev in q["expected_evidence"] if ev["figure_id"] is not None]
            text_targets = [ev for ev in q["expected_evidence"] if ev["figure_id"] is None]
            assert figure_targets, f"{q['query_id']}: hybrid query missing figure target"
            assert text_targets, f"{q['query_id']}: hybrid query missing text target"


# ---------------------------------------------------------------------------
# Duplicate canonical figure IDs — only intended multi-crop cases allowed
# ---------------------------------------------------------------------------


def test_duplicate_figure_ids_are_intentional_multi_crop(manifest: dict) -> None:
    """A figure_id may map to multiple manifest entries only when they share the
    same (document_id, page_number) — i.e. one conceptual figure spanning several
    crops (the v1 precedent: Figure 2's two sub-images share one figure_id).
    A figure_id spanning different pages would indicate a grounding error.
    """
    from collections import defaultdict

    locations: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for doc in manifest["documents"]:
        for fig in doc.get("figures", []):
            locations[fig["figure_id"]].add((doc["document_id"], fig["page_number"]))

    for fid, locs in locations.items():
        assert len(locs) == 1, f"figure_id {fid!r} spans multiple locations: {locs}"


# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------


def test_dataset_version_present(dataset: dict) -> None:
    assert dataset["dataset_version"] == "v2.0.0"


def test_corpus_version_present(dataset: dict) -> None:
    assert dataset["corpus_version"] == "v2.0.0"


def test_query_count_field_matches_actual(dataset: dict, queries: list[dict]) -> None:
    assert dataset["query_count"] == len(queries)
