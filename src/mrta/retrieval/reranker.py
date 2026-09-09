"""mrta.retrieval.reranker — cross-encoder reranking.

Two rerankers live here
-----------------------
``Reranker`` (Stage 4, production) re-scores ``Chunk`` lists inside
``rag_query``. It returns plain chunks and keeps no score provenance, which is
all the production path needs.

``CrossEncoderReranker`` (PR5) re-scores ``FusedCandidate`` lists produced by
canonical RRF fusion. It returns ``RerankedCandidate`` objects that carry the
RRF score and RRF rank *alongside* the cross-encoder score, because PR5's whole
question is how the two rankings differ. Overwriting ``candidate.score`` with
the cross-encoder score would destroy exactly the diagnostic being measured, so
``FusedCandidate`` is never mutated — it is held by reference.

Both wrap the same sentence-transformers CrossEncoder. They are kept separate
because they consume different types and answer different questions; the
production path is deliberately untouched by PR5. This mirrors the two-API
arrangement in ``mrta.retrieval.fusion``.

Scope limitation
----------------
``cross-encoder/ms-marco-MiniLM-L-6-v2`` is a TEXT cross-encoder. It cannot
inspect image pixels. Figure candidates are reranked through their
production-derived *textual* representation only. PR5 is therefore a
query-aware textual reranker over multimodal retrieval candidates, NOT an
image-text multimodal reranker. Failure on visually specific queries may
reflect the representation rather than the reranker's ranking quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from mrta.core.schemas import Chunk
from mrta.retrieval.fusion import EVIDENCE_TYPE_TEXT, FusedCandidate

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


class Reranker:
    """Re-scores retrieved chunks using a cross-encoder model.

    Loads the cross-encoder lazily on first instantiation. The model is
    downloaded from HuggingFace Hub on first use — mock in tests to avoid
    network access.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self._model: CrossEncoder = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(self, query: str, chunks: list[Chunk], top_n: int = 3) -> list[Chunk]:
        """Return top_n chunks sorted by cross-encoder relevance score (descending).

        If top_n exceeds len(chunks), all chunks are returned in score order.
        """
        if not chunks:
            return []
        pairs = [(query, c.text) for c in chunks]
        scores = self._model.predict(pairs)  # type: ignore[arg-type]
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:top_n]]


# ---------------------------------------------------------------------------
# PR5 — query-aware cross-encoder reranking over fused multimodal candidates
# ---------------------------------------------------------------------------

# Payload keys the evaluation runner writes onto FusedCandidate.payload. These
# are the ONLY channel through which candidate text reaches the reranker, which
# is what keeps benchmark ground truth (source_note, expected_evidence,
# retrieval_challenge, difficulty, canonical figure names) structurally unable
# to leak into model input: none of those fields is ever written to a payload.
PAYLOAD_CHUNK_TEXT = "chunk_text"
PAYLOAD_FIGURE_CAPTION = "figure_caption"  # VLM-generated
PAYLOAD_FIGURE_EXTRACTED_CAPTION = "figure_extracted_caption"
PAYLOAD_FIGURE_DESCRIPTION = "figure_description"  # VLM-generated
PAYLOAD_FIGURE_NEARBY_TEXT = "figure_nearby_text"  # page-extraction fallback

# reranker_text_source provenance labels (PR5 spec section 8).
TEXT_SOURCE_CHUNK = "chunk_text"
TEXT_SOURCE_VLM_CAPTION = "vlm_caption"
TEXT_SOURCE_EXTRACTED_CAPTION = "extracted_caption"
TEXT_SOURCE_NEARBY_FALLBACK = "nearby_text_fallback"
TEXT_SOURCE_COMBINED = "combined_production_text"
TEXT_SOURCE_EMPTY = "empty"


@dataclass(frozen=True)
class RerankedCandidate:
    """One candidate after cross-encoder scoring, with both scores preserved.

    ``candidate`` is the original FusedCandidate, held by reference and never
    mutated, so ``candidate.score`` remains the RRF score. ``original_rrf_score``
    and ``original_rrf_rank`` are copied out so an artifact row can be written
    without reaching back into the fused list, and so the pre/post rank movement
    PR5 measures survives serialization.
    """

    candidate: FusedCandidate

    reranker_score: float
    reranker_rank: int

    original_rrf_score: float
    original_rrf_rank: int

    reranker_text: str
    reranker_text_source: str


def _resolve_reranker_text(candidate: FusedCandidate) -> tuple[str, str]:
    """Return (text, provenance_label) for one fused candidate.

    Deterministic and side-effect free: the same candidate always yields the
    same pair, and the candidate is only read from.

    Text evidence uses the actual retrieved chunk text — never page-level text,
    which would be a coarser representation than the one that was retrieved.

    Figure evidence uses production-derived figure text in the priority order
    fixed by the PR5 spec (VLM caption > extracted caption > nearby-text
    fallback). When several legitimate production fields are present they are
    combined under explicit labels rather than discarded: every field is
    length-bounded upstream (nearby_text is capped at ~400 characters at
    extraction time), so the combined string stays well inside the model's
    512-token window and no field can silently truncate another away.

    No new text is generated here. Nothing derived from the benchmark — query
    annotations, expected evidence, canonical figure names — is read.
    """
    payload = candidate.payload

    if candidate.evidence_type == EVIDENCE_TYPE_TEXT:
        chunk_text = (payload.get(PAYLOAD_CHUNK_TEXT) or "").strip()
        if not chunk_text:
            return "", TEXT_SOURCE_EMPTY
        return chunk_text, TEXT_SOURCE_CHUNK

    caption = (payload.get(PAYLOAD_FIGURE_CAPTION) or "").strip()
    extracted = (payload.get(PAYLOAD_FIGURE_EXTRACTED_CAPTION) or "").strip()
    description = (payload.get(PAYLOAD_FIGURE_DESCRIPTION) or "").strip()
    nearby = (payload.get(PAYLOAD_FIGURE_NEARBY_TEXT) or "").strip()

    parts: list[str] = []
    if caption:
        parts.append(f"Figure caption: {caption}")
    elif extracted:
        parts.append(f"Figure caption: {extracted}")
    if description:
        parts.append(f"Description: {description}")
    if nearby:
        parts.append(f"Context: {nearby}")

    if not parts:
        # A figure with no production text at all. Scored as an empty string so
        # the outcome stays deterministic instead of raising mid-evaluation.
        return "", TEXT_SOURCE_EMPTY

    # Provenance describes what the model actually saw. A single field keeps its
    # own label; anything combined is reported as combined_production_text, and
    # in the v2 corpus that label implies a VLM caption was present, because
    # nearby_text is the only field a fallback figure has.
    if len(parts) > 1:
        source = TEXT_SOURCE_COMBINED
    elif caption:
        source = TEXT_SOURCE_VLM_CAPTION
    elif extracted:
        source = TEXT_SOURCE_EXTRACTED_CAPTION
    elif description:
        source = TEXT_SOURCE_VLM_CAPTION
    else:
        source = TEXT_SOURCE_NEARBY_FALLBACK

    return "\n".join(parts), source


def candidate_to_reranker_text(candidate: FusedCandidate) -> str:
    """Deterministic textual representation the cross-encoder scores."""
    return _resolve_reranker_text(candidate)[0]


def candidate_text_source(candidate: FusedCandidate) -> str:
    """Provenance label for the text ``candidate_to_reranker_text`` returns."""
    return _resolve_reranker_text(candidate)[1]


class _ScoredEntry(NamedTuple):
    """One candidate mid-rerank: its incoming RRF position plus its new score.

    A typed NamedTuple rather than a plain dict — a dict literal mixing a
    FusedCandidate, a float, an int and two strs collapses to dict[str, object]
    under mypy, which erases every field's type at the point of use.
    """

    candidate: FusedCandidate
    score: float
    rrf_rank: int
    text: str
    text_source: str


class CrossEncoderReranker:
    """Query-aware cross-encoder reranking over canonical fused candidates.

    RRF orders the candidate set entering this stage; the cross-encoder is the
    final ranking stage. The two scores are never blended — PR5 measures what
    relevance-aware reranking does to a rank-fusion ordering, and a blend would
    confound that.
    """

    def __init__(self, model_name: str | None = None) -> None:
        """Load the cross-encoder. Downloads from HuggingFace Hub on first use.

        Args:
            model_name: overrides the pinned ``settings.reranker_model_name``.
                Present for tests and reproducibility, not for model sweeps.
        """
        from mrta.core.config import settings

        resolved = model_name or settings.reranker_model_name
        from sentence_transformers import CrossEncoder

        self._model: CrossEncoder = CrossEncoder(resolved)
        self.model_name = resolved

    def rerank(
        self,
        query: str,
        candidates: list[FusedCandidate],
        top_k: int = 5,
    ) -> list[RerankedCandidate]:
        """Re-score fused candidates by query relevance and return the top_k.

        List position is authoritative for the incoming RRF rank, matching the
        convention ``reciprocal_rank_fusion_canonical`` uses: index 0 is rank 1.
        A candidate's own attributes are never written to.

        All pairs are scored in a single ``predict`` call so batching is the
        model's to optimize.

        Ties are broken deterministically by RRF rank ascending, then
        canonical_id ascending. Cross-encoder scores tie routinely on short or
        empty figure text, so an unspecified order here would make repeated
        runs disagree.
        """
        if not candidates:
            return []

        resolved = [_resolve_reranker_text(c) for c in candidates]
        pairs = [(query, text) for text, _ in resolved]
        scores = self._model.predict(pairs)  # type: ignore[arg-type]

        entries = [
            _ScoredEntry(
                candidate=candidate,
                score=float(score),
                rrf_rank=position + 1,
                text=text,
                text_source=text_source,
            )
            for position, (candidate, score, (text, text_source)) in enumerate(
                zip(candidates, scores, resolved)
            )
        ]

        entries.sort(key=lambda e: (-e.score, e.rrf_rank, e.candidate.canonical_id))

        return [
            RerankedCandidate(
                candidate=e.candidate,
                reranker_score=e.score,
                reranker_rank=i + 1,
                original_rrf_score=e.candidate.score,
                original_rrf_rank=e.rrf_rank,
                reranker_text=e.text,
                reranker_text_source=e.text_source,
            )
            for i, e in enumerate(entries[:top_k])
        ]
