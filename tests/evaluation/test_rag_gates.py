"""Regression gate tests for retrieval metrics.

Loads delta-based thresholds from baselines/retrieval_metrics.json.
Each gate constructs a "perfect" mock retrieval so the score must meet
the baseline. The real regression value is the baseline itself; tolerance
absorbs natural variation.

TODO (per gate): Replace perfect_retrieved with VectorStore.search(question, k=N)
output after PDFs are indexed. The assertion logic stays identical.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mrta.evaluation.metrics import citation_coverage, mrr, ndcg_at_k, recall_at_k
from tests.evaluation.conftest import load_golden_qa

BASELINES_PATH = Path(__file__).parent / "baselines" / "retrieval_metrics.json"
GOLDEN_QA_PATH = Path(__file__).parent / "datasets" / "golden_qa.yaml"


@pytest.fixture(scope="module")
def baselines() -> dict:
    return json.loads(BASELINES_PATH.read_text())


@pytest.fixture(scope="module")
def qa_items() -> list[dict]:
    return load_golden_qa(GOLDEN_QA_PATH)


def _perfect_retrieved(qa_items: list[dict], k: int) -> tuple[list[str], list[str]]:
    """Build a flat list of (retrieved, expected) where every expected doc is placed at the top."""
    all_retrieved: list[str] = []
    all_expected: list[str] = []
    for item in qa_items:
        expected = item["expected_documents"]
        if not expected:
            continue
        # perfect mock: expected docs at positions 1..len(expected), then padding
        retrieved = list(expected) + [f"pad_{i}.pdf" for i in range(max(k - len(expected), 0))]
        all_retrieved.extend(retrieved[:k])
        all_expected.extend(expected)
    return all_retrieved, all_expected


def test_baselines_file_structure(baselines: dict) -> None:
    for key, entry in baselines.items():
        assert "baseline" in entry, f"{key} missing 'baseline'"
        assert "tolerance" in entry, f"{key} missing 'tolerance'"


def test_recall_at_5_gate(baselines: dict, qa_items: list[dict]) -> None:
    # TODO: replace perfect_retrieved with VectorStore.search(q, k=5) after indexing
    entry = baselines["recall_at_5"]
    scores = []
    for item in qa_items:
        expected = item["expected_documents"]
        if not expected:
            continue
        retrieved = list(expected) + [f"pad_{i}.pdf" for i in range(5)]
        scores.append(recall_at_k(retrieved, expected, k=5))
    score = sum(scores) / len(scores)
    assert score >= entry["baseline"] - entry["tolerance"]


def test_recall_at_10_gate(baselines: dict, qa_items: list[dict]) -> None:
    # TODO: replace perfect_retrieved with VectorStore.search(q, k=10) after indexing
    entry = baselines["recall_at_10"]
    scores = []
    for item in qa_items:
        expected = item["expected_documents"]
        if not expected:
            continue
        retrieved = list(expected) + [f"pad_{i}.pdf" for i in range(10)]
        scores.append(recall_at_k(retrieved, expected, k=10))
    score = sum(scores) / len(scores)
    assert score >= entry["baseline"] - entry["tolerance"]


def test_mrr_gate(baselines: dict, qa_items: list[dict]) -> None:
    # TODO: replace perfect_retrieved with VectorStore.search(q, k=10) after indexing
    entry = baselines["mrr"]
    scores = []
    for item in qa_items:
        expected = item["expected_documents"]
        if not expected:
            continue
        retrieved = list(expected) + ["pad.pdf"]
        scores.append(mrr(retrieved, expected))
    score = sum(scores) / len(scores)
    assert score >= entry["baseline"] - entry["tolerance"]


def test_ndcg_at_10_gate(baselines: dict, qa_items: list[dict]) -> None:
    # TODO: replace perfect_retrieved with VectorStore.search(q, k=10) after indexing
    entry = baselines["ndcg_at_10"]
    scores = []
    for item in qa_items:
        expected = item["expected_documents"]
        if not expected:
            continue
        retrieved = list(expected) + [f"pad_{i}.pdf" for i in range(10)]
        scores.append(ndcg_at_k(retrieved, expected, k=10))
    score = sum(scores) / len(scores)
    assert score >= entry["baseline"] - entry["tolerance"]


def test_citation_coverage_gate(baselines: dict, qa_items: list[dict]) -> None:
    # TODO: replace perfect_retrieved with VectorStore.search(q, k=10) after indexing
    entry = baselines["citation_coverage"]
    scores = []
    for item in qa_items:
        expected = item["expected_documents"]
        if not expected:
            continue
        retrieved = list(expected) + ["pad.pdf"]
        scores.append(citation_coverage(retrieved, expected))
    score = sum(scores) / len(scores)
    assert score >= entry["baseline"] - entry["tolerance"]
