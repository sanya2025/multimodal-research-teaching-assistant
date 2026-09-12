"""mrta.retrieval.canonical_pipeline — production multimodal retrieval orchestration.

Wires the PR1-PR5 retrieval stack into one production entry point::

    Text ──────┐
    Caption ───┼──► canonical RRF ──► top-20 ──► CrossEncoder ──► top-5
    CLIP ──────┘

This module *orchestrates*; it does not reimplement retrieval, fusion or
ranking. Fusion is ``reciprocal_rank_fusion_canonical`` (PR4) and reranking is
``CrossEncoderReranker`` (PR5), both used unmodified.

Why a production adapter exists
-------------------------------
``mrta.eval.adapter.EvalAdapter`` resolves a figure's canonical ``figure_id`` by
looking it up in the frozen v2 benchmark manifest. Production has no manifest,
and production code must not depend on evaluation artifacts. But it does not
need one: ingestion already assigns every figure the stable identity
``(doc_id, page, figure_index)`` — that triple is exactly what
``FigureRecord.to_evidence_record()`` encodes into ``evidence_id``. So a
canonical figure id can be *derived* here with no benchmark dependency, and
PR4's ``canonical_identity()`` then works unchanged.

That is what makes the caption stream and the CLIP stream collapse onto one
candidate when they return the same physical figure, which is the property the
legacy ``evidence_id`` fusion path cannot provide across differing record types.

Parameters are fixed at the PR4/PR5 evaluated values (pool 20, rrf_k 60, top-5).
They are not tuned here — see ADR-007 and results/v2/pr5_reranker_metrics.json.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from mrta.core.schemas import Chunk, EvidenceRecord, VisualRecord
from mrta.eval.types import CanonicalEvidence, RetrievedCandidate
from mrta.observability.tracing import trace_span
from mrta.retrieval.fusion import canonical_identity, reciprocal_rank_fusion_canonical
from mrta.retrieval.reranker import (
    PAYLOAD_CHUNK_TEXT,
    PAYLOAD_FIGURE_CAPTION,
    PAYLOAD_FIGURE_DESCRIPTION,
    PAYLOAD_FIGURE_NEARBY_TEXT,
    RerankedCandidate,
)

if TYPE_CHECKING:
    from mrta.retrieval.reranker import CrossEncoderReranker

# The PR4/PR5 evaluated configuration. Not tuned in production.
CANDIDATE_POOL_SIZE = 20
RRF_K = 60
RERANK_TOP_N = 20
FINAL_TOP_K = 5

# Stream names, fixed so fusion's sorted-stream determinism is stable.
STREAM_TEXT = "text"
STREAM_CAPTION = "caption"
STREAM_CLIP = "clip"

# Citation metadata carried through fusion on FusedCandidate.payload.
# FusedCandidate holds document_id but not the human-facing PDF filename or the
# figure's image asset, and citations need both. These keys are metadata only:
# the reranker's text resolver reads none of them, so they cannot reach model
# input as content.
PAYLOAD_SOURCE = "source"
PAYLOAD_EVIDENCE_ID = "evidence_id"
PAYLOAD_MODALITY = "modality"
PAYLOAD_FIGURE_INDEX = "figure_index"
PAYLOAD_IMAGE_PATH = "image_path"

# reranker_text_source value recorded when the reranker did not run.
TEXT_SOURCE_NOT_RERANKED = "not_reranked"


class _TextStore(Protocol):
    """Minimal text-retrieval surface used here (VectorStore satisfies it)."""

    def search_with_scores(self, query: str, k: int = ...) -> list[tuple[Chunk, float]]: ...


def production_figure_id(page: int, figure_index: int | None) -> str:
    """Derive a stable canonical figure id from production ingestion metadata.

    ``(doc_id, page, figure_index)`` already uniquely identifies a physical
    figure in this corpus; ``document_id`` is carried separately by
    ``CanonicalEvidence``, so the id only needs to disambiguate within a
    document. A ``figure_index`` of None (a whole-page visual) maps to index 0,
    matching the ingestion convention that page renders use index 0.

    Deliberately *not* the benchmark's semantic ids ("fig_transformer_arch") —
    those live only in the frozen eval manifest.
    """
    return f"p{page}_f{figure_index if figure_index is not None else 0}"


def chunk_to_candidate(chunk: Chunk, score: float, rank: int) -> RetrievedCandidate:
    """Map a retrieved text Chunk to canonical form (figure_id is always None)."""
    return RetrievedCandidate(
        candidate_id=chunk.chunk_id,
        evidence=CanonicalEvidence(
            document_id=chunk.doc_id,
            page_number=chunk.page,
            figure_id=None,
        ),
        score=score,
        rank=rank,
    )


def evidence_record_to_candidate(
    record: EvidenceRecord,
    score: float,
    rank: int,
) -> RetrievedCandidate:
    """Map a caption/visual EvidenceRecord to canonical form.

    Text-modality records keep ``figure_id=None`` so they can never merge with
    figure evidence on the same page.
    """
    figure_id = (
        None
        if record.modality == "text"
        else production_figure_id(record.page, record.figure_index)
    )
    return RetrievedCandidate(
        candidate_id=record.evidence_id,
        evidence=CanonicalEvidence(
            document_id=record.doc_id,
            page_number=record.page,
            figure_id=figure_id,
        ),
        score=score,
        rank=rank,
    )


def visual_record_to_candidate(
    record: VisualRecord,
    score: float,
    rank: int,
) -> RetrievedCandidate:
    """Map an ImageStore VisualRecord to canonical form.

    VisualRecord persists its own ``figure_id``. It is deliberately ignored in
    favour of the derived id so that a CLIP hit and a caption hit on the same
    physical figure produce the *same* canonical identity and therefore merge.
    """
    return RetrievedCandidate(
        candidate_id=record.record_id,
        evidence=CanonicalEvidence(
            document_id=record.document_id,
            page_number=record.page,
            figure_id=production_figure_id(record.page, record.figure_index),
        ),
        score=score,
        rank=rank,
    )


@dataclass
class RetrievalDiagnostics:
    """Per-stage timings and stream availability for one multimodal query."""

    stream_sizes: dict[str, int] = field(default_factory=dict)
    degraded_streams: dict[str, str] = field(default_factory=dict)
    latency_text: float = 0.0
    latency_caption: float = 0.0
    latency_clip: float = 0.0
    latency_fusion: float = 0.0
    latency_rerank: float = 0.0
    reranker_used: bool = False
    fused_pool_size: int = 0


def _text_payload(chunk: Chunk) -> dict[str, Any]:
    """Reranker text plus citation metadata for one retrieved chunk."""
    return {
        PAYLOAD_CHUNK_TEXT: chunk.text,
        PAYLOAD_SOURCE: chunk.source,
        PAYLOAD_EVIDENCE_ID: chunk.chunk_id,
        PAYLOAD_MODALITY: "text",
    }


def _figure_payload(record: Any) -> dict[str, Any]:
    """Reranker text plus citation metadata for one figure candidate.

    Handles both record types production may hold: ``EvidenceRecord`` (caption
    store, and the legacy ``VisualVectorStore``) and ``VisualRecord``
    (``ImageStore``). A VisualRecord carries no text of its own, so a figure
    found only by CLIP contributes its image asset here and takes its textual
    representation from the caption stream's payload when fusion merges them.
    """
    if isinstance(record, VisualRecord):
        return {
            PAYLOAD_SOURCE: record.document_id,
            PAYLOAD_EVIDENCE_ID: record.record_id,
            PAYLOAD_MODALITY: "image",
            PAYLOAD_FIGURE_INDEX: record.figure_index,
            PAYLOAD_IMAGE_PATH: record.image_path,
        }
    return {
        PAYLOAD_FIGURE_CAPTION: record.caption,
        PAYLOAD_FIGURE_DESCRIPTION: record.detailed_description,
        PAYLOAD_FIGURE_NEARBY_TEXT: record.nearby_text,
        PAYLOAD_SOURCE: record.source,
        PAYLOAD_EVIDENCE_ID: record.evidence_id,
        PAYLOAD_MODALITY: record.modality,
        PAYLOAD_FIGURE_INDEX: record.figure_index,
        PAYLOAD_IMAGE_PATH: record.image_path,
    }


def retrieve_multimodal(
    query: str,
    text_store: _TextStore,
    caption_store: Any | None = None,
    image_store: Any | None = None,
    reranker: CrossEncoderReranker | None = None,
    candidate_pool_size: int = CANDIDATE_POOL_SIZE,
    rrf_k: int = RRF_K,
    top_k: int = FINAL_TOP_K,
) -> tuple[list[RerankedCandidate], RetrievalDiagnostics]:
    """Run the validated production retrieval stack for one query.

    Stages: per-stream retrieval → canonical conversion → canonical RRF →
    top-20 pool → cross-encoder rerank → top-5.

    Graceful degradation (spec section 8): text retrieval is required, because
    it is the only stream the production ingestion path always builds. Caption
    and CLIP are optional; if either is absent or raises, the query proceeds
    without it and the reason is recorded in the diagnostics and the trace span.
    If the reranker is absent or raises, the canonical RRF ordering is returned
    instead, wrapped so callers always receive the same result type.

    Returns:
        (final candidates in rank order, diagnostics)
    """
    diagnostics = RetrievalDiagnostics()

    with trace_span(
        "mrta.canonical_pipeline.retrieve",
        {
            "retrieval.fusion_method": "rrf_canonical",
            "retrieval.rrf_k": rrf_k,
            "retrieval.candidate_pool_size": candidate_pool_size,
            "retrieval.final_top_k": top_k,
        },
    ) as span:
        streams: dict[str, list[RetrievedCandidate]] = {}
        payloads: dict[str, dict[str, dict[str, Any]]] = {}

        # --- text stream (required) ---
        t0 = time.perf_counter()
        text_hits = text_store.search_with_scores(query, k=candidate_pool_size)
        diagnostics.latency_text = time.perf_counter() - t0

        text_candidates = [
            chunk_to_candidate(chunk, score, i + 1) for i, (chunk, score) in enumerate(text_hits)
        ]
        if text_candidates:
            streams[STREAM_TEXT] = text_candidates
            payloads[STREAM_TEXT] = {
                "|".join(canonical_identity(cand)): _text_payload(chunk)
                for (chunk, _), cand in zip(text_hits, text_candidates)
            }
        diagnostics.stream_sizes[STREAM_TEXT] = len(text_candidates)

        # --- caption stream (optional) ---
        if caption_store is not None:
            t0 = time.perf_counter()
            try:
                caption_hits = caption_store.search_with_scores(query, k=candidate_pool_size)
                caption_candidates = [
                    evidence_record_to_candidate(rec, score, i + 1)
                    for i, (rec, score) in enumerate(caption_hits)
                ]
                if caption_candidates:
                    streams[STREAM_CAPTION] = caption_candidates
                    payloads[STREAM_CAPTION] = {
                        "|".join(canonical_identity(cand)): _figure_payload(rec)
                        for (rec, _), cand in zip(caption_hits, caption_candidates)
                    }
                diagnostics.stream_sizes[STREAM_CAPTION] = len(caption_candidates)
            except Exception as exc:  # noqa: BLE001 — degrade, but never silently
                diagnostics.degraded_streams[STREAM_CAPTION] = f"{type(exc).__name__}: {exc}"
            finally:
                diagnostics.latency_caption = time.perf_counter() - t0

        # --- CLIP stream (optional) ---
        if image_store is not None:
            t0 = time.perf_counter()
            try:
                clip_hits = _search_image_store(image_store, query, candidate_pool_size)
                clip_candidates = [
                    _visual_hit_to_candidate(rec, score, i + 1)
                    for i, (rec, score) in enumerate(clip_hits)
                ]
                if clip_candidates:
                    streams[STREAM_CLIP] = clip_candidates
                    payloads[STREAM_CLIP] = {
                        "|".join(canonical_identity(cand)): _figure_payload(rec)
                        for (rec, _), cand in zip(clip_hits, clip_candidates)
                    }
                diagnostics.stream_sizes[STREAM_CLIP] = len(clip_candidates)
            except Exception as exc:  # noqa: BLE001 — degrade, but never silently
                diagnostics.degraded_streams[STREAM_CLIP] = f"{type(exc).__name__}: {exc}"
            finally:
                diagnostics.latency_clip = time.perf_counter() - t0

        if not streams:
            span.set_attribute("retrieval.final_candidates", 0)
            return [], diagnostics

        # --- canonical RRF (PR4, unmodified) ---
        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion_canonical(streams, k=rrf_k, top_k=None, payloads=payloads)
        diagnostics.latency_fusion = time.perf_counter() - t0

        pool = fused[:RERANK_TOP_N]
        diagnostics.fused_pool_size = len(pool)

        # --- cross-encoder reranking (PR5, unmodified) ---
        final: list[RerankedCandidate]
        if reranker is not None and pool:
            t0 = time.perf_counter()
            try:
                final = reranker.rerank(query, pool, top_k=top_k)
                diagnostics.reranker_used = True
            except Exception as exc:  # noqa: BLE001 — fall back to RRF ordering
                diagnostics.degraded_streams["reranker"] = f"{type(exc).__name__}: {exc}"
                final = _wrap_rrf_ordering(pool[:top_k])
            finally:
                diagnostics.latency_rerank = time.perf_counter() - t0
        else:
            final = _wrap_rrf_ordering(pool[:top_k])

        span.set_attribute(
            "retrieval.text_candidates", diagnostics.stream_sizes.get(STREAM_TEXT, 0)
        )
        span.set_attribute(
            "retrieval.caption_candidates", diagnostics.stream_sizes.get(STREAM_CAPTION, 0)
        )
        span.set_attribute(
            "retrieval.clip_candidates", diagnostics.stream_sizes.get(STREAM_CLIP, 0)
        )
        span.set_attribute("retrieval.fused_pool_size", diagnostics.fused_pool_size)
        span.set_attribute("retrieval.final_candidates", len(final))
        span.set_attribute("retrieval.reranker_used", diagnostics.reranker_used)
        span.set_attribute("retrieval.degraded_streams", sorted(diagnostics.degraded_streams))
        span.set_attribute("latency.text_retrieval", round(diagnostics.latency_text, 4))
        span.set_attribute("latency.caption_retrieval", round(diagnostics.latency_caption, 4))
        span.set_attribute("latency.clip_retrieval", round(diagnostics.latency_clip, 4))
        span.set_attribute("latency.fusion", round(diagnostics.latency_fusion, 4))
        span.set_attribute("latency.reranking", round(diagnostics.latency_rerank, 4))

    return final, diagnostics


def _search_image_store(image_store: Any, query: str, k: int) -> list[tuple[Any, float]]:
    """Search either an ImageStore (top_k=) or a VisualVectorStore (k=)."""
    if hasattr(image_store, "search_with_scores"):
        return list(image_store.search_with_scores(query, k=k))
    return list(image_store.search(query, top_k=k))


def _visual_hit_to_candidate(record: Any, score: float, rank: int) -> RetrievedCandidate:
    """Convert whichever visual record type the configured store returned."""
    if isinstance(record, VisualRecord):
        return visual_record_to_candidate(record, score, rank)
    return evidence_record_to_candidate(record, score, rank)


def _wrap_rrf_ordering(candidates: list) -> list[RerankedCandidate]:
    """Present an unreranked RRF ordering as RerankedCandidates.

    ``reranker_score`` mirrors the RRF score purely so the sequence stays
    orderable by the same field; the RRF score and rank remain available
    separately, so no caller can mistake one for a cross-encoder score.
    ``reranker_text_source`` records that no reranking happened.
    """
    from mrta.retrieval.reranker import candidate_to_reranker_text

    return [
        RerankedCandidate(
            candidate=candidate,
            reranker_score=candidate.score,
            reranker_rank=i + 1,
            original_rrf_score=candidate.score,
            original_rrf_rank=i + 1,
            reranker_text=candidate_to_reranker_text(candidate),
            reranker_text_source=TEXT_SOURCE_NOT_RERANKED,
        )
        for i, candidate in enumerate(candidates)
    ]
