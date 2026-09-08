"""mrta.eval.retrieval_metrics — retrieval quality metrics over CanonicalEvidence.

All metrics operate on RetrievedCandidate lists (already ranked by score)
and CanonicalEvidence target lists.

Deduplication guarantee: if multiple retrieved candidates map to the same
canonical evidence key (same document_id, page_number, figure_id), only the
highest-ranked occurrence contributes to DCG / recall / hit. This prevents
inflating nDCG > 1.0 when many chunks from the same page are retrieved.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from mrta.eval.types import CanonicalEvidence, RetrievedCandidate


def is_hit(
    candidates: Sequence[RetrievedCandidate],
    targets: Sequence[CanonicalEvidence],
    k: int,
) -> bool:
    """True if at least one target evidence appears in the top-k candidates.

    Empty targets → True (nothing required, nothing missing).
    """
    if not targets:
        return True
    top_k = list(candidates)[:k]
    for candidate in top_k:
        for target in targets:
            if target.matches(candidate.evidence):
                return True
    return False


def recall_at_k(
    candidates: Sequence[RetrievedCandidate],
    targets: Sequence[CanonicalEvidence],
    k: int,
) -> float:
    """Fraction of targets found in the top-k candidates (after deduplication).

    Empty targets → 1.0 (nothing to recall).
    Duplicates in candidates that map to the same target are counted once.
    """
    if not targets:
        return 1.0

    top_k = list(candidates)[:k]
    seen_keys: set[tuple] = set()
    matched_target_indices: set[int] = set()

    for candidate in top_k:
        key = candidate.evidence.key()
        if key in seen_keys:
            continue
        seen_keys.add(key)
        for i, target in enumerate(targets):
            if i not in matched_target_indices and target.matches(candidate.evidence):
                matched_target_indices.add(i)

    return len(matched_target_indices) / len(targets)


def hit_rate_at_k(
    candidates: Sequence[RetrievedCandidate],
    targets: Sequence[CanonicalEvidence],
    k: int,
) -> float:
    """1.0 if at least one target is in top-k, else 0.0."""
    return 1.0 if is_hit(candidates, targets, k) else 0.0


def mean_reciprocal_rank(
    candidates: Sequence[RetrievedCandidate],
    targets: Sequence[CanonicalEvidence],
) -> float:
    """Reciprocal rank of the first candidate that matches any target.

    Returns 0.0 if no target is found anywhere in the candidate list.
    """
    for i, candidate in enumerate(candidates, start=1):
        for target in targets:
            if target.matches(candidate.evidence):
                return 1.0 / i
    return 0.0


def ndcg_at_k(
    candidates: Sequence[RetrievedCandidate],
    targets: Sequence[CanonicalEvidence],
    k: int,
) -> float:
    """nDCG@k with binary relevance and deduplication.

    Deduplication: once a canonical evidence key is seen, subsequent candidates
    with the same key contribute rel=0 — preventing nDCG from exceeding 1.0
    when many chunks map to the same page/figure.

    Each distinct target can only be matched once (matched_target_indices tracks
    which targets have already been credited).

    Empty targets → 1.0.
    """
    if not targets:
        return 1.0

    top_k = list(candidates)[:k]
    seen_keys: set[tuple] = set()
    matched_target_indices: set[int] = set()
    relevance: list[int] = []

    for candidate in top_k:
        key = candidate.evidence.key()
        if key in seen_keys:
            relevance.append(0)
            continue
        seen_keys.add(key)

        rel = 0
        for i, target in enumerate(targets):
            if i not in matched_target_indices and target.matches(candidate.evidence):
                matched_target_indices.add(i)
                rel = 1
                break
        relevance.append(rel)

    dcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevance))
    n_relevant = min(len(targets), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(n_relevant))

    return dcg / idcg if idcg > 0 else 1.0
