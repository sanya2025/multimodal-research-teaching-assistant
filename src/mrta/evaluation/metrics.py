"""mrta.evaluation.metrics — deterministic RAG quality metrics."""

from __future__ import annotations

import re

from mrta.core.schemas import Chunk


def answer_relevance(question: str, answer: str) -> float:
    """Score in [0,1]: fraction of non-trivial question keywords found in the answer."""
    stop = {
        "what",
        "is",
        "the",
        "a",
        "an",
        "how",
        "why",
        "does",
        "do",
        "did",
        "are",
        "in",
        "of",
        "to",
        "and",
        "for",
        "on",
        "at",
        "by",
    }
    tokens = {
        t.lower().strip(".,?!;:") for t in question.split() if t.lower().strip(".,?!;:") not in stop
    }
    if not tokens:
        return 1.0
    answer_lower = answer.lower()
    return sum(1 for t in tokens if t in answer_lower) / len(tokens)


def faithfulness(answer: str, chunks: list[Chunk]) -> float:
    """Score in [0,1]: fraction of answer sentences grounded in at least one chunk."""
    sentences = [s.strip() for s in re.split(r"[.!?]+", answer) if s.strip()]
    if not sentences:
        return 1.0
    context = " ".join(c.text.lower() for c in chunks)
    grounded = 0
    for sent in sentences:
        tokens = {t.lower().strip(".,!?;:") for t in sent.split() if len(t) > 3}
        if not tokens or any(t in context for t in tokens):
            grounded += 1
    return grounded / len(sentences)


def citation_correctness(answer: str, chunks: list[Chunk]) -> float:
    """Score in [0,1]: fraction of [page N] citations that refer to a page in chunks."""
    cited = [int(m.group(1)) for m in re.finditer(r"\[page (\d+)", answer, re.IGNORECASE)]
    if not cited:
        return 1.0
    valid_pages = {c.page for c in chunks}
    return sum(1 for p in cited if p in valid_pages) / len(cited)


def hallucination_rate(answer: str, chunks: list[Chunk]) -> float:
    """Fraction of answer sentences with no grounding chunk (1 - faithfulness)."""
    return 1.0 - faithfulness(answer, chunks)


__all__ = ["answer_relevance", "faithfulness", "citation_correctness", "hallucination_rate"]
