"""mrta.retrieval.fusion — Reciprocal Rank Fusion for multimodal evidence.

Fuses any number of independently-ranked EvidenceRecord lists using RRF:

    RRF(d) = Σ 1 / (k + rank_r(d))

where rank_r(d) is the 1-based position of document d in ranked list r,
and k (default 60) controls the smoothing.

Scores from sentence-transformers and CLIP are not directly comparable
(different embedding spaces, different score calibrations). RRF operates
on rank order only, side-stepping score incompatibility entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mrta.core.schemas import EvidenceRecord


@dataclass
class FusedResult:
    """A single deduplicated evidence item produced by rank fusion.

    Attributes
    ----------
    record:
        The canonical EvidenceRecord. When the same evidence_id appears in
        multiple input lists, the copy from the highest-ranked list is kept.
    rrf_score:
        Sum of 1/(k + rank_r(d)) across all input lists.
        Higher is better. Records absent from a list contribute 0 for that list.
    per_list_rank:
        Mapping from list name to 1-based rank in that list.
        A list is absent from the dict if the record was not in it.
    source_modality:
        Taken directly from record.modality — "text", "image", or "page".
    """

    record: EvidenceRecord
    rrf_score: float
    per_list_rank: dict[str, int] = field(default_factory=dict)
    source_modality: str = "text"


def reciprocal_rank_fusion(
    named_lists: dict[str, list[EvidenceRecord]],
    k: int = 60,
    top_n: int | None = None,
) -> list[FusedResult]:
    """Fuse ranked evidence lists using Reciprocal Rank Fusion.

    Parameters
    ----------
    named_lists:
        Dict mapping a list name (e.g. "text", "caption", "visual") to a
        ranked list of EvidenceRecords. Lists are assumed to be in rank order:
        index 0 is rank 1.
    k:
        Smoothing constant (default 60). Higher k reduces the weight advantage
        of top ranks.
    top_n:
        If set, return only the top_n results by RRF score. Returns all results
        when None.

    Returns
    -------
    list[FusedResult]
        Deduplicated, sorted by rrf_score descending.
    """
    scores: dict[str, float] = {}
    per_list_ranks: dict[str, dict[str, int]] = {}
    canonical: dict[str, EvidenceRecord] = {}
    best_rank: dict[str, int] = {}

    for list_name, records in named_lists.items():
        for rank_0, record in enumerate(records):
            eid = record.evidence_id
            rank_1 = rank_0 + 1
            scores[eid] = scores.get(eid, 0.0) + 1.0 / (k + rank_1)
            if eid not in per_list_ranks:
                per_list_ranks[eid] = {}
            per_list_ranks[eid][list_name] = rank_1
            # keep the canonical record from the list where it ranked highest
            if eid not in best_rank or rank_1 < best_rank[eid]:
                best_rank[eid] = rank_1
                canonical[eid] = record

    results = [
        FusedResult(
            record=canonical[eid],
            rrf_score=score,
            per_list_rank=per_list_ranks[eid],
            source_modality=canonical[eid].modality,
        )
        for eid, score in scores.items()
    ]

    results.sort(key=lambda r: r.rrf_score, reverse=True)
    if top_n is not None:
        results = results[:top_n]
    return results
