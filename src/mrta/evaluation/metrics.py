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


def recall_at_k(retrieved_sources: list[str], expected_docs: list[str], k: int) -> float:
    """Fraction of expected_docs found in the first k entries of retrieved_sources."""
    top = retrieved_sources[:k]
    if not expected_docs:
        return 1.0
    return sum(1 for d in expected_docs if d in top) / len(expected_docs)


def mrr(retrieved_sources: list[str], expected_docs: list[str]) -> float:
    """Reciprocal rank of the first retrieved_source that is in expected_docs."""
    expected_set = set(expected_docs)
    for rank, src in enumerate(retrieved_sources, start=1):
        if src in expected_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_sources: list[str], expected_docs: list[str], k: int = 10) -> float:
    """nDCG@k — rewards placing expected_docs near the top of retrieved_sources."""
    import math

    expected_set = set(expected_docs)
    top = retrieved_sources[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, src in enumerate(top, start=1) if src in expected_set
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(expected_docs), k) + 1))
    return dcg / ideal if ideal > 0 else 1.0


def citation_coverage(retrieved_sources: list[str], expected_docs: list[str]) -> float:
    """Retrieval-side: fraction of expected_docs found anywhere in retrieved_sources."""
    if not expected_docs:
        return 1.0
    retrieved_set = set(retrieved_sources)
    return sum(1 for d in expected_docs if d in retrieved_set) / len(expected_docs)


__all__ = [
    "answer_relevance",
    "citation_correctness",
    "citation_coverage",
    "faithfulness",
    "hallucination_rate",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
]
