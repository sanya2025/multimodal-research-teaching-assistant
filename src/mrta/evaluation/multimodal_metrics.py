"""mrta.evaluation.multimodal_metrics — retrieval and citation metrics for multimodal RAG.

Extends the text-only metrics in metrics.py with figure-aware recall and
multimodal citation correctness checks for [T#]/[V#] labelled answers.
"""

from __future__ import annotations

import re

from mrta.core.schemas import EvidenceRecord, MultimodalAnswer


def figure_recall_at_k(
    retrieved: list[EvidenceRecord],
    expected_figures: list[dict],
    k: int,
) -> float:
    """Fraction of expected figures found in the top-k retrieved visual records.

    A retrieved record matches an expected figure when (source, page, figure_index)
    all match. Records with figure_index=None match expected entries that also have
    figure_index=None (page-level visual evidence).

    Args:
        retrieved: Full ordered list of retrieved EvidenceRecords (all modalities).
        expected_figures: List of dicts with keys ``source``, ``page``,
            ``figure_index`` (int or None).
        k: Cut-off rank.

    Returns:
        Float in [0, 1]. Returns 1.0 when expected_figures is empty.
    """
    if not expected_figures:
        return 1.0

    top_k_visual = [r for r in retrieved[:k] if r.modality in ("image", "page")]
    retrieved_keys = {(r.source, r.page, r.figure_index) for r in top_k_visual}
    hits = sum(
        1
        for fig in expected_figures
        if (fig["source"], fig["page"], fig.get("figure_index")) in retrieved_keys
    )
    return hits / len(expected_figures)


def multimodal_recall_at_k(
    retrieved: list[EvidenceRecord],
    expected_text_pages: list[int],
    expected_figures: list[dict],
    k: int,
    source: str = "",
) -> dict[str, float]:
    """Text and visual recall at k, reported separately and as a combined score.

    Text recall: fraction of expected pages present in the top-k text records.
    Visual recall: ``figure_recall_at_k`` on the top-k visual records.
    Overall: geometric mean of text and visual recall (0 if either is 0).

    Keeping modality-specific scores prevents aggregate numbers from hiding
    failures in one modality (e.g. perfect text recall masking zero visual recall).

    Args:
        retrieved: Full ordered list of EvidenceRecords.
        expected_text_pages: Pages that should appear in text evidence.
        expected_figures: Figure dicts (source, page, figure_index) that should
            be retrieved as visual evidence.
        k: Cut-off rank applied to each modality slice separately.
        source: Optional document name filter for text-page matching.

    Returns:
        Dict with keys ``text``, ``visual``, ``overall``, each a float in [0, 1].
    """
    top_k = retrieved[:k]

    # text recall
    if not expected_text_pages:
        text_recall = 1.0
    else:
        text_records = [r for r in top_k if r.modality == "text"]
        if source:
            text_records = [r for r in text_records if r.source == source]
        retrieved_pages = {r.page for r in text_records}
        hits = sum(1 for p in expected_text_pages if p in retrieved_pages)
        text_recall = hits / len(expected_text_pages)

    visual_recall = figure_recall_at_k(retrieved, expected_figures, k=k)

    # geometric mean so that either zero pulls overall to zero
    import math

    overall = math.sqrt(text_recall * visual_recall)

    return {"text": text_recall, "visual": visual_recall, "overall": overall}


def multimodal_citation_correctness(
    answer: MultimodalAnswer,
    retrieved: list[EvidenceRecord],
) -> dict[str, float]:
    """Measure citation quality in a MultimodalAnswer at three levels.

    format_score
        Fraction of [T#] and [V#] labels in the answer text that are correctly
        formatted (match the regex ``\\[T\\d+\\]`` or ``\\[V\\d+\\]``).

    provenance_score
        Fraction of cited [T#]/[V#] labels that map to a real citation in
        ``answer.text_citations`` or ``answer.visual_citations``.

    support_score
        Fraction of citation labels whose cited source/page appears in the
        retrieved evidence list (proxy for whether the citation is grounded).

    overall
        Mean of the three scores.

    Args:
        answer: A ``MultimodalAnswer`` returned by ``MultimodalRAG.ask()``.
        retrieved: The evidence list the answer was generated from.

    Returns:
        Dict with keys ``format``, ``provenance``, ``support``, ``overall``.
    """
    text_in_answer = answer.answer

    # ── format correctness ──────────────────────────────────────────────────
    raw_refs = re.findall(r"\[(?:T|V)\d+\]", text_in_answer, re.IGNORECASE)
    valid_refs = re.findall(r"\[(?:T|V)\d+\]", text_in_answer)
    format_score = len(valid_refs) / len(raw_refs) if raw_refs else 1.0

    # ── provenance correctness ──────────────────────────────────────────────
    all_labels = {c.label for c in answer.text_citations + answer.visual_citations}
    cited_labels = set(valid_refs)
    if not cited_labels:
        provenance_score = 1.0
    else:
        mapped = sum(1 for label in cited_labels if label in all_labels)
        provenance_score = mapped / len(cited_labels)

    # ── support correctness ─────────────────────────────────────────────────
    retrieved_keys: set[tuple[str, int]] = {(r.source, r.page) for r in retrieved}
    all_citations = answer.text_citations + answer.visual_citations
    cited_in_answer = [c for c in all_citations if c.label in cited_labels]
    if not cited_in_answer:
        support_score = 1.0
    else:
        supported = sum(1 for c in cited_in_answer if (c.source, c.page) in retrieved_keys)
        support_score = supported / len(cited_in_answer)

    overall = (format_score + provenance_score + support_score) / 3
    return {
        "format": format_score,
        "provenance": provenance_score,
        "support": support_score,
        "overall": overall,
    }


__all__ = [
    "figure_recall_at_k",
    "multimodal_citation_correctness",
    "multimodal_recall_at_k",
]
