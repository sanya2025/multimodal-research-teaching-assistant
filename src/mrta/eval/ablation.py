"""mrta.eval.ablation — frozen ablation matrix over the MRTA retrieval stack.

Evaluation-only orchestration. Executes each configuration against the frozen
benchmark, scores retrieval and (optionally) generation, and attributes every
failure to a specific stage.

    query dataset
        ↓
    AblationConfig (stream set, fusion on/off, reranking on/off)
        ↓
    retrieval  →  canonical RRF  →  CrossEncoder
        ↓
    retrieval metrics          (mrta.eval.retrieval_metrics)
        ↓
    generation (optional)
        ↓
    citation + grounding metrics (mrta.eval.generation_metrics)
        ↓
    aggregation

Nothing here reimplements retrieval, fusion or ranking: single-stream configs
score the stream's own ranking directly, and fused configs call PR4's
``reciprocal_rank_fusion_canonical`` and PR5's ``CrossEncoderReranker``
unmodified. That is what makes the historical configurations comparable with
their frozen PR1-PR5 results.

Why single-stream configs bypass fusion
---------------------------------------
RRF over one list is a monotone transform of that list's ranking, so it cannot
change the ordering — but routing a single stream through fusion would still
change its *candidate identity* handling. PR1-PR3 scored the raw stream, so the
ablation does too, and the historical-reproduction check in the report verifies
that choice against the frozen numbers rather than assuming it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from mrta.eval.types import CanonicalEvidence, RetrievedCandidate

# The evaluated PR4/PR5 constants. PR7 does not tune; it measures.
DEFAULT_CANDIDATE_DEPTH = 20
DEFAULT_RRF_K = 60
DEFAULT_FINAL_TOP_K = 5

STREAM_TEXT = "text"
STREAM_CAPTION = "caption"
STREAM_CLIP = "clip"

# Failure attribution categories (spec section 21). Ordered from earliest stage
# to latest, because a query is attributed to the first stage that failed.
FAILURE_RETRIEVAL_MISS = "retrieval_miss"  # target never entered any candidate pool
FAILURE_RANKING_MISS = "ranking_miss"  # target in pool but below final cutoff
FAILURE_CITATION_MISSING = "citation_missing"  # supplied to generator, not cited
FAILURE_CITATION_RELEVANCE = "citation_relevance"  # cited supplied-but-wrong evidence
FAILURE_CITATION_VALIDITY = "citation_validity"  # cited evidence never supplied
FAILURE_ANSWER_SUPPORT = "answer_support"  # answer weakly supported by context
FAILURE_NONE = "none"

STATUS_OK = "ok"
STATUS_ERROR = "error"


@dataclass(frozen=True)
class AblationConfig:
    """One frozen point in the ablation matrix.

    ``streams`` is the set of retrieval streams contributing candidates.
    ``fuse`` applies canonical RRF; it is meaningless for a single stream and is
    rejected rather than silently ignored. ``rerank`` applies the cross-encoder.
    ``oracle_evidence`` bypasses retrieval entirely and hands the generator the
    benchmark's ground-truth evidence — an evaluation-only upper bound, never a
    production configuration.
    """

    config_id: str
    streams: tuple[str, ...] = ()
    fuse: bool = False
    rerank: bool = False
    oracle_evidence: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if self.oracle_evidence:
            if self.streams or self.fuse or self.rerank:
                raise ValueError(
                    f"{self.config_id}: oracle_evidence bypasses retrieval and cannot "
                    "combine with streams, fusion or reranking"
                )
            return
        if not self.streams:
            raise ValueError(f"{self.config_id}: at least one stream is required")
        for stream in self.streams:
            if stream not in (STREAM_TEXT, STREAM_CAPTION, STREAM_CLIP):
                raise ValueError(f"{self.config_id}: unknown stream {stream!r}")
        if self.fuse and len(self.streams) < 2:
            raise ValueError(
                f"{self.config_id}: fusion needs 2+ streams; a single stream fused "
                "with itself is just that stream's own ranking"
            )
        if self.rerank and not self.fuse:
            raise ValueError(
                f"{self.config_id}: reranking operates on a fused candidate pool "
                "(PR5 reranks the RRF pool, not a raw stream)"
            )

    @property
    def is_retrieval(self) -> bool:
        return not self.oracle_evidence


# The frozen matrix (spec section 7 + section 8). Declared before any PR7 result
# was seen; post-hoc additions must be labelled exploratory.
FROZEN_CONFIGURATIONS: tuple[AblationConfig, ...] = (
    AblationConfig("text_only", (STREAM_TEXT,), description="PR1 text baseline"),
    AblationConfig("caption_only", (STREAM_CAPTION,), description="PR2 caption baseline"),
    AblationConfig("clip_only", (STREAM_CLIP,), description="PR3 CLIP baseline"),
    AblationConfig(
        "text_caption_rrf", (STREAM_TEXT, STREAM_CAPTION), fuse=True, description="PR4 T+C"
    ),
    AblationConfig(
        "text_clip_rrf", (STREAM_TEXT, STREAM_CLIP), fuse=True, description="PR4 T+CLIP"
    ),
    AblationConfig(
        "caption_clip_rrf", (STREAM_CAPTION, STREAM_CLIP), fuse=True, description="PR4 C+CLIP"
    ),
    AblationConfig(
        "text_caption_clip_rrf",
        (STREAM_TEXT, STREAM_CAPTION, STREAM_CLIP),
        fuse=True,
        description="PR4 three-stream",
    ),
    AblationConfig(
        "text_caption_rrf_reranked",
        (STREAM_TEXT, STREAM_CAPTION),
        fuse=True,
        rerank=True,
        description="PR5 T+C -> CE",
    ),
    AblationConfig(
        "full_reranked",
        (STREAM_TEXT, STREAM_CAPTION, STREAM_CLIP),
        fuse=True,
        rerank=True,
        description="PR5/PR6 full system",
    ),
    AblationConfig(
        "oracle_evidence_generation",
        oracle_evidence=True,
        description="Evaluation-only upper bound: ground-truth evidence to the generator",
    ),
)

CONFIGURATIONS_BY_ID: dict[str, AblationConfig] = {c.config_id: c for c in FROZEN_CONFIGURATIONS}

# Configurations whose retrieval numbers should reproduce a frozen PR1-PR5 result.
HISTORICAL_CONFIGURATIONS: tuple[str, ...] = (
    "text_only",
    "caption_only",
    "clip_only",
    "text_caption_rrf",
    "text_caption_clip_rrf",
    "text_caption_rrf_reranked",
    "full_reranked",
)


@dataclass
class QueryResult:
    """One (query x configuration) row."""

    query_id: str
    config_id: str
    document_id: str | None = None
    intent: str | None = None
    difficulty: str | None = None
    retrieval_challenge: str | None = None

    status: str = STATUS_OK
    error_type: str | None = None
    error_message: str | None = None

    expected_evidence: list[dict] = field(default_factory=list)
    final_evidence: list[dict] = field(default_factory=list)
    pool_target_rank: int | None = None
    final_target_rank: int | None = None

    retrieval_metrics: dict = field(default_factory=dict)
    generation_metrics: dict = field(default_factory=dict)
    coverage: dict = field(default_factory=dict)

    generated_answer: str | None = None
    resolved_citations: list[dict] = field(default_factory=list)

    # Provenance of the first figure in the FINAL SLICE — i.e. what the system
    # actually surfaced. Named explicitly because it is not the same population
    # as the target-figure provenance below, and conflating them invites a false
    # comparison with PR5.
    top_retrieved_figure_provenance: str | None = None
    # Provenance of the figure the benchmark EXPECTS. This is PR5's grouping and
    # the one comparable with its measurements.
    target_figure_provenance: str | None = None
    failure_category: str = FAILURE_NONE

    latency_ms: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "config_id": self.config_id,
            "document_id": self.document_id,
            "intent": self.intent,
            "difficulty": self.difficulty,
            "retrieval_challenge": self.retrieval_challenge,
            "status": self.status,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "expected_evidence": self.expected_evidence,
            "final_evidence": self.final_evidence,
            "pool_target_rank": self.pool_target_rank,
            "final_target_rank": self.final_target_rank,
            "retrieval_metrics": self.retrieval_metrics,
            "generation_metrics": self.generation_metrics,
            "coverage": self.coverage,
            "generated_answer": self.generated_answer,
            "resolved_citations": self.resolved_citations,
            "top_retrieved_figure_provenance": self.top_retrieved_figure_provenance,
            "target_figure_provenance": self.target_figure_provenance,
            "failure_category": self.failure_category,
            "latency_ms": self.latency_ms,
        }


def parse_targets(query: dict) -> list[CanonicalEvidence]:
    """Canonical targets for one benchmark query.

    Accepts both schemas: v2 uses ``expected_evidence``, v1 ``target_evidence``.
    """
    raw = query.get("expected_evidence", query.get("target_evidence", []))
    return [
        CanonicalEvidence(
            document_id=t["document_id"],
            page_number=t["page_number"],
            figure_id=t.get("figure_id"),
        )
        for t in raw
    ]


def evidence_as_dicts(evidence: Sequence[CanonicalEvidence]) -> list[dict]:
    return [
        {
            "document_id": e.document_id,
            "page_number": e.page_number,
            "figure_id": e.figure_id,
        }
        for e in evidence
    ]


def target_rank(
    candidates: Sequence[RetrievedCandidate],
    targets: Sequence[CanonicalEvidence],
    figures_only: bool = False,
) -> int | None:
    """1-based rank of the first candidate matching any target, else None."""
    wanted = [t for t in targets if t.figure_id is not None] if figures_only else list(targets)
    for position, candidate in enumerate(candidates, start=1):
        if any(t.matches(candidate.evidence) for t in wanted):
            return position
    return None


def compute_retrieval_metrics(
    candidates: Sequence[RetrievedCandidate],
    targets: Sequence[CanonicalEvidence],
    final_top_k: int = DEFAULT_FINAL_TOP_K,
) -> dict:
    """The eight PR7 retrieval metrics, computed on the top-k slice.

    The slice is not cosmetic: ``mean_reciprocal_rank`` takes no k and scans
    whatever list it is handed, so passing a depth-20 pool would silently yield
    MRR@20 and break comparability with every frozen PR1-PR5 number. PR4 guards
    this the same way at its call site.
    """
    from mrta.eval.retrieval_metrics import (
        figure_recall_at_k,
        hit_rate_at_k,
        mean_reciprocal_rank,
        ndcg_at_k,
        recall_at_k,
    )

    top_k = list(candidates)[:final_top_k]
    has_figures = any(t.figure_id is not None for t in targets)

    return {
        "recall_at_1": round(recall_at_k(top_k, targets, 1), 6),
        "recall_at_5": round(recall_at_k(top_k, targets, final_top_k), 6),
        "hit_at_1": round(hit_rate_at_k(top_k, targets, 1), 6),
        "hit_at_5": round(hit_rate_at_k(top_k, targets, final_top_k), 6),
        "mrr_at_5": round(mean_reciprocal_rank(top_k, targets), 6),
        "ndcg_at_5": round(ndcg_at_k(top_k, targets, final_top_k), 6),
        "figure_recall_at_1": (
            round(figure_recall_at_k(top_k, targets, 1), 6) if has_figures else None
        ),
        "figure_recall_at_5": (
            round(figure_recall_at_k(top_k, targets, final_top_k), 6) if has_figures else None
        ),
        "both_target_recall_at_5": _hybrid_both_covered(top_k, targets, final_top_k),
    }


def _hybrid_both_covered(
    candidates: Sequence[RetrievedCandidate],
    targets: Sequence[CanonicalEvidence],
    k: int,
) -> bool | None:
    """True when both a text and a figure target appear in the top-k.

    None when the query lacks one of the two kinds, so it is reported as N/A
    rather than counted as a success — the PR4 convention.
    """
    figure_targets = [t for t in targets if t.figure_id is not None]
    text_targets = [t for t in targets if t.figure_id is None]
    if not figure_targets or not text_targets:
        return None
    top = list(candidates)[:k]
    figure_hit = any(t.matches(c.evidence) for c in top for t in figure_targets)
    text_hit = any(t.matches(c.evidence) for c in top for t in text_targets)
    return figure_hit and text_hit


def classify_failure(
    *,
    targets: Sequence[CanonicalEvidence],
    pool_rank: int | None,
    final_rank: int | None,
    generation_ran: bool,
    citation_scores: dict | None,
    validity_rate: float | None,
    support_score: float | None,
    support_threshold: float = 0.5,
) -> str:
    """Attribute a query to the earliest stage that failed.

    Earliest-stage attribution matters: if the target never reached the
    generator, a missing citation is not the generator's fault, and counting it
    as one would make the generation stage look worse than it is.
    """
    if not targets:
        return FAILURE_NONE

    if pool_rank is None:
        return FAILURE_RETRIEVAL_MISS
    if final_rank is None:
        return FAILURE_RANKING_MISS

    if not generation_ran:
        return FAILURE_NONE

    if validity_rate is not None and validity_rate < 1.0:
        return FAILURE_CITATION_VALIDITY
    if citation_scores:
        if citation_scores.get("citation_recall", 1.0) < 1.0:
            return FAILURE_CITATION_MISSING
        if citation_scores.get("citation_precision", 1.0) < 1.0:
            return FAILURE_CITATION_RELEVANCE
    if support_score is not None and support_score < support_threshold:
        return FAILURE_ANSWER_SUPPORT
    return FAILURE_NONE


def _mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return round(sum(present) / len(present), 6) if present else None


def _fraction_true(values: Sequence[bool | None]) -> float | None:
    present = [v for v in values if v is not None]
    return round(sum(1 for v in present if v) / len(present), 6) if present else None


def aggregate(rows: Sequence[QueryResult]) -> dict:
    """Average a set of per-query rows.

    Only successful rows contribute to metric means; failed rows are counted
    separately so an execution error can never quietly improve an average by
    dropping a hard query from the denominator.
    """
    successful = [r for r in rows if r.status == STATUS_OK]
    failed = [r for r in rows if r.status != STATUS_OK]

    summary: dict = {
        "sample_count": len(rows),
        "successful_count": len(successful),
        "failed_count": len(failed),
        "failure_rate": round(len(failed) / len(rows), 6) if rows else 0.0,
    }

    if not successful:
        return summary

    for metric in (
        "recall_at_1",
        "recall_at_5",
        "hit_at_1",
        "hit_at_5",
        "mrr_at_5",
        "ndcg_at_5",
        "figure_recall_at_1",
        "figure_recall_at_5",
    ):
        summary[metric] = _mean([r.retrieval_metrics.get(metric) for r in successful])

    summary["both_target_recall_at_5"] = _fraction_true(
        [r.retrieval_metrics.get("both_target_recall_at_5") for r in successful]
    )

    generated = [r for r in successful if r.generated_answer is not None]
    if generated:
        for metric in (
            "citation_precision",
            "citation_recall",
            "citation_f1",
            "citation_validity_rate",
            "hallucinated_citation_rate",
            "lexical_support_score",
            "supported_claim_fraction",
            "unsupported_claim_fraction",
            "context_token_count",
            "answer_token_count",
        ):
            summary[metric] = _mean([r.generation_metrics.get(metric) for r in generated])

        for key in ("text_target_covered", "figure_target_covered", "both_targets_covered"):
            summary[key] = _fraction_true([r.coverage.get(key) for r in generated])
        summary["any_target_covered"] = _fraction_true(
            [r.coverage.get("any_target_covered") for r in generated]
        )
        summary["generated_count"] = len(generated)

    for stage in ("retrieval", "fusion", "rerank", "generation", "total"):
        summary[f"latency_{stage}_ms"] = _mean([r.latency_ms.get(stage) for r in successful])

    counts: dict[str, int] = {}
    for row in successful:
        counts[row.failure_category] = counts.get(row.failure_category, 0) + 1
    summary["failure_categories"] = dict(sorted(counts.items()))

    return summary


def percentile(values: Sequence[float], pct: float) -> float | None:
    """Nearest-rank percentile. Small samples make interpolation false precision."""
    present = sorted(v for v in values if v is not None)
    if not present:
        return None
    index = max(0, math.ceil(pct / 100 * len(present)) - 1)
    return round(present[index], 6)


def figure_provenance(payload: dict[str, Any] | None) -> str | None:
    """Derive a figure's textual provenance from its production-derived fields.

    Derived rather than read from ``caption_source`` because the frozen v2
    caption index predates that field: every record there carries None, so
    reading it would report "unknown" for the entire benchmark. PR5 derived the
    same way, which keeps the provenance breakdown comparable with its numbers.
    """
    if not payload:
        return None
    if payload.get("figure_caption") or payload.get("figure_description"):
        return "vlm_caption"
    if payload.get("figure_extracted_caption"):
        return "extracted_caption"
    if payload.get("figure_nearby_text"):
        return "nearby_text_fallback"
    return None


__all__ = [
    "AblationConfig",
    "CONFIGURATIONS_BY_ID",
    "FROZEN_CONFIGURATIONS",
    "HISTORICAL_CONFIGURATIONS",
    "QueryResult",
    "aggregate",
    "classify_failure",
    "compute_retrieval_metrics",
    "evidence_as_dicts",
    "figure_provenance",
    "parse_targets",
    "percentile",
    "target_rank",
]
