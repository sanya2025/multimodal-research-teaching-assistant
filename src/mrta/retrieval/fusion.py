"""mrta.retrieval.fusion — Reciprocal Rank Fusion for multimodal evidence.

Fuses any number of independently-ranked lists using RRF:

    RRF(d) = Σ 1 / (k + rank_r(d))

where rank_r(d) is the 1-based position of document d in ranked list r,
and k (default 60) controls the smoothing.

Scores from sentence-transformers and CLIP are not directly comparable
(different embedding spaces, different score calibrations). RRF operates
on rank order only, side-stepping score incompatibility entirely.

Two fusion APIs live here
--------------------------
``reciprocal_rank_fusion`` (Stage 4, production) fuses ``EvidenceRecord``
lists keyed by ``evidence_id``. It is consumed by ``MultimodalRetriever``.

``reciprocal_rank_fusion_canonical`` (PR4) fuses ``RetrievedCandidate``
lists keyed by *canonical evidence identity*, so the same physical figure
retrieved by both the caption and CLIP streams collapses into one candidate
that accumulates both streams' RRF contributions. ``evidence_id`` keying
cannot do this: a caption record and a CLIP record for the same figure carry
different ids. The PR4 path also guarantees deterministic tie-breaking, which
matters because RRF produces ties frequently.

Both are kept because they answer different questions and have different
callers; the production path is deliberately untouched by PR4.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from mrta.core.schemas import EvidenceRecord
from mrta.eval.types import RetrievedCandidate


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


# ---------------------------------------------------------------------------
# PR4 — canonical-identity fusion
# ---------------------------------------------------------------------------

# Evidence-type tags used in canonical fusion identities.
EVIDENCE_TYPE_TEXT = "text"
EVIDENCE_TYPE_FIGURE = "figure"


@dataclass(frozen=True)
class FusedCandidate:
    """One deduplicated evidence item produced by canonical rank fusion.

    ``canonical_id`` is the deduplication key. Two retrieved items collapse into
    one FusedCandidate exactly when their canonical ids match, which is what lets
    the caption and CLIP streams agree on a physical figure and have their RRF
    contributions summed.

    Frozen so a fused ranking cannot be mutated after scoring. ``source_ranks``,
    ``source_scores`` and ``payload`` are ordinary dicts, so treat them as
    read-only by convention.
    """

    canonical_id: str
    evidence_type: str

    document_id: str
    page: int

    figure_id: str | None = None
    chunk_id: str | None = None

    score: float = 0.0

    modality_sources: tuple[str, ...] = ()
    source_ranks: dict[str, int] = field(default_factory=dict)

    # Diagnostics only — never used for scoring or tie-breaking.
    source_scores: dict[str, float] = field(default_factory=dict)

    payload: dict[str, Any] = field(default_factory=dict)

    def best_rank(self) -> int:
        """Best (lowest) rank this candidate achieved in any contributing stream."""
        return min(self.source_ranks.values()) if self.source_ranks else 0


def canonical_identity(candidate: RetrievedCandidate) -> tuple[str, str, str]:
    """Canonical fusion identity for a RetrievedCandidate.

    Figure evidence -> ("figure", document_id, "p{page}:{figure_id}")
    Text evidence   -> ("text",   document_id, candidate_id)

    Why not reuse CanonicalEvidence.key() directly: two text chunks on the same
    page share the identical key ``(doc, page, None)``, so keying on it would
    wrongly merge distinct text evidence. For text the stream's own
    ``candidate_id`` (the chunk_id, globally stable because doc_id is a content
    hash) is the correct identity. Figures keep page+figure_id so that the same
    physical figure found by different streams — and multiple crops of one
    conceptual figure — collapse together.
    """
    ev = candidate.evidence
    if ev.figure_id is not None:
        return (EVIDENCE_TYPE_FIGURE, ev.document_id, f"p{ev.page_number}:{ev.figure_id}")
    return (EVIDENCE_TYPE_TEXT, ev.document_id, candidate.candidate_id)


def _merge_payload(base: dict[str, Any], incoming: Mapping[str, Any]) -> dict[str, Any]:
    """Merge stream payloads deterministically.

    First writer wins for any key already present. Streams are processed in
    sorted stream-name order by the caller, so the result does not depend on
    the ordering of the input dict — only on stream names, which are fixed.
    Payload never influences the RRF score.
    """
    merged = dict(base)
    for key in sorted(incoming):
        if key not in merged or merged[key] is None:
            value = incoming[key]
            if value is not None:
                merged[key] = value
    return merged


def reciprocal_rank_fusion_canonical(
    streams: Mapping[str, Sequence[RetrievedCandidate]],
    k: int = 60,
    top_k: int | None = None,
    payloads: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
) -> list[FusedCandidate]:
    """Fuse per-stream ranked candidate lists by canonical evidence identity.

    Args:
        streams: stream name -> candidates in rank order (index 0 is rank 1).
            A candidate's own ``rank`` attribute is ignored; list position is
            authoritative, so callers cannot desynchronise ranks from ordering.
        k: RRF smoothing constant. 60 is the standard default and the value
            PR4 fixes; other values exist only to let tests verify the formula.
        top_k: truncate the fused ranking to this many candidates. None keeps all.
        payloads: optional stream name -> canonical_id -> payload dict, merged
            into the fused candidate for diagnostics. Never affects scoring.

    Returns:
        Fused candidates sorted by the deterministic PR4 ordering:
        score desc, best source rank asc, stream count desc, canonical_id asc.

    A candidate absent from a stream contributes exactly nothing for that stream
    — no penalty, no synthetic bottom rank. If the same canonical identity appears
    more than once within a single stream (e.g. several crops of one figure), only
    its first — best-ranked — occurrence contributes, so one stream can never
    inflate a candidate by listing it repeatedly.
    """
    if k <= 0:
        raise ValueError(f"rrf k must be positive, got {k}")

    accum: dict[tuple[str, str, str], dict[str, Any]] = {}

    # Sorted stream order makes payload merging and dict construction independent
    # of the caller's dict ordering; RRF summation is order-independent anyway.
    for stream_name in sorted(streams):
        seen_in_stream: set[tuple[str, str, str]] = set()
        for position, candidate in enumerate(streams[stream_name]):
            identity = canonical_identity(candidate)
            if identity in seen_in_stream:
                continue  # first occurrence in a stream is the best-ranked one
            seen_in_stream.add(identity)

            rank = position + 1  # ranks are 1-indexed
            entry = accum.get(identity)
            if entry is None:
                ev = candidate.evidence
                entry = {
                    "evidence_type": identity[0],
                    "document_id": ev.document_id,
                    "page": ev.page_number,
                    "figure_id": ev.figure_id,
                    "chunk_id": (candidate.candidate_id if ev.figure_id is None else None),
                    "score": 0.0,
                    "sources": [],
                    "source_ranks": {},
                    "source_scores": {},
                    "payload": {},
                }
                accum[identity] = entry

            entry["score"] += 1.0 / (k + rank)
            entry["sources"].append(stream_name)
            entry["source_ranks"][stream_name] = rank
            entry["source_scores"][stream_name] = candidate.score

            if payloads is not None:
                stream_payloads = payloads.get(stream_name)
                if stream_payloads is not None:
                    incoming = stream_payloads.get(_canonical_id_str(identity))
                    if incoming:
                        entry["payload"] = _merge_payload(entry["payload"], incoming)

    fused = [
        FusedCandidate(
            canonical_id=_canonical_id_str(identity),
            evidence_type=entry["evidence_type"],
            document_id=entry["document_id"],
            page=entry["page"],
            figure_id=entry["figure_id"],
            chunk_id=entry["chunk_id"],
            score=entry["score"],
            modality_sources=tuple(entry["sources"]),
            source_ranks=dict(entry["source_ranks"]),
            source_scores=dict(entry["source_scores"]),
            payload=dict(entry["payload"]),
        )
        for identity, entry in accum.items()
    ]

    fused.sort(
        key=lambda c: (
            -c.score,
            c.best_rank(),
            -len(c.modality_sources),
            c.canonical_id,
        )
    )

    if top_k is not None:
        fused = fused[:top_k]
    return fused


def _canonical_id_str(identity: tuple[str, str, str]) -> str:
    """Render a canonical identity tuple as a stable string id."""
    return "|".join(identity)
