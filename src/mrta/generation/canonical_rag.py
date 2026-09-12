"""mrta.generation.canonical_rag — generation over canonical multimodal evidence.

Consumes the output of ``mrta.retrieval.canonical_pipeline.retrieve_multimodal``
(the PR4 canonical RRF + PR5 cross-encoder stack) and turns it into a grounded
answer with verified text and figure citations.

    RerankedCandidate[]  →  evidence views  →  prompt + images  →  VLM
                                           →  citations  →  verification

Kept separate from ``MultimodalRAG`` (Stage 7) because the two consume different
evidence types: ``MultimodalRAG`` takes ``EvidenceRecord`` lists from the legacy
``evidence_id``-keyed retriever, while this takes ``RerankedCandidate`` carrying
canonical identity and full ranking provenance. The Stage-7 path is untouched.

Citation integrity
------------------
Labels are assigned by the application, never parsed out of model output. The
model is told to reference ``[T1]``/``[F1]``; the application then resolves those
labels back to the evidence it actually retrieved. A label the model invents
resolves to nothing and is reported rather than trusted, so a fabricated page,
figure id or file path cannot reach the response.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mrta.core.exceptions import LLMError
from mrta.core.schemas import MultimodalAnswer, MultimodalCitation
from mrta.observability.tracing import trace_span
from mrta.prompts import load_prompt
from mrta.retrieval.canonical_pipeline import (
    PAYLOAD_EVIDENCE_ID,
    PAYLOAD_FIGURE_INDEX,
    PAYLOAD_IMAGE_PATH,
    PAYLOAD_MODALITY,
    PAYLOAD_SOURCE,
    RetrievalDiagnostics,
    retrieve_multimodal,
)
from mrta.retrieval.fusion import EVIDENCE_TYPE_FIGURE
from mrta.retrieval.reranker import (
    PAYLOAD_CHUNK_TEXT,
    PAYLOAD_FIGURE_CAPTION,
    PAYLOAD_FIGURE_DESCRIPTION,
    PAYLOAD_FIGURE_NEARBY_TEXT,
    RerankedCandidate,
)

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

# Roots an emitted image_path is allowed to live under, relative to the working
# directory. Anything outside these is treated as untrusted and dropped, so a
# malformed index can never turn into an arbitrary host path in an API response.
ALLOWED_IMAGE_ROOTS: tuple[str, ...] = ("data",)


@dataclass
class EvidenceView:
    """One retrieved candidate normalized for prompting and citation.

    ``label`` is assigned by the application ("[T1]", "[F1]"), which is what
    makes citation verification possible: the model can only legitimately
    reference a label that appears here.
    """

    label: str
    evidence_type: str
    document_id: str
    source: str
    page: int
    evidence_id: str
    text: str = ""
    figure_id: str | None = None
    chunk_id: str | None = None
    figure_index: int | None = None
    caption: str | None = None
    image_path: str | None = None
    modality: str = "text"
    modality_sources: tuple[str, ...] = ()
    rrf_rank: int | None = None
    reranker_rank: int | None = None

    def to_citation(self) -> MultimodalCitation:
        """Structured provenance for this evidence, for the API response."""
        return MultimodalCitation(
            label=self.label,
            evidence_id=self.evidence_id,
            modality=self.modality,  # type: ignore[arg-type]
            source=self.source,
            page=self.page,
            figure_index=self.figure_index,
            evidence_type=self.evidence_type,  # type: ignore[arg-type]
            document_id=self.document_id,
            figure_id=self.figure_id,
            chunk_id=self.chunk_id,
            image_path=self.image_path,
            caption=self.caption,
            modality_sources=list(self.modality_sources),
            rrf_rank=self.rrf_rank,
            reranker_rank=self.reranker_rank,
        )


@dataclass
class CitationVerification:
    """Which labels the answer referenced, and which of those were invented."""

    referenced: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.unknown


def safe_image_path(raw: str | None) -> str | None:
    """Return ``raw`` only if it is a plausible, existing MRTA asset path.

    Guards the response against three distinct failure modes: an index that
    recorded an absolute host path, a path that escapes the asset tree via
    ``..``, and a path whose file has since been deleted. Any of those yields
    None, and the figure keeps its textual evidence without an image reference.
    """
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_absolute():
        return None
    try:
        resolved = (Path.cwd() / candidate).resolve()
        resolved.relative_to(Path.cwd().resolve())
    except (ValueError, OSError):
        return None
    if not any(part == root for part in candidate.parts for root in ALLOWED_IMAGE_ROOTS):
        return None
    if not resolved.exists():
        return None
    return raw


def build_evidence_views(candidates: list[RerankedCandidate]) -> list[EvidenceView]:
    """Normalize reranked candidates into labelled text and figure evidence.

    Text evidence is labelled [T1..], figure evidence [F1..]; the two are kept
    semantically distinct rather than flattened into one block, and the final
    ranking order within each kind is preserved.
    """
    views: list[EvidenceView] = []
    n_text = n_figure = 0

    for reranked in candidates:
        fused = reranked.candidate
        payload = fused.payload
        is_figure = fused.evidence_type == EVIDENCE_TYPE_FIGURE

        source = str(payload.get(PAYLOAD_SOURCE) or fused.document_id)
        evidence_id = str(payload.get(PAYLOAD_EVIDENCE_ID) or fused.canonical_id)

        if is_figure:
            n_figure += 1
            label = f"[F{n_figure}]"
            caption = _figure_description(payload)
            views.append(
                EvidenceView(
                    label=label,
                    evidence_type="figure",
                    document_id=fused.document_id,
                    source=source,
                    page=fused.page,
                    evidence_id=evidence_id,
                    text=caption or "",
                    figure_id=fused.figure_id,
                    figure_index=payload.get(PAYLOAD_FIGURE_INDEX),
                    caption=caption,
                    image_path=safe_image_path(payload.get(PAYLOAD_IMAGE_PATH)),
                    modality=str(payload.get(PAYLOAD_MODALITY) or "image"),
                    modality_sources=fused.modality_sources,
                    rrf_rank=reranked.original_rrf_rank,
                    reranker_rank=reranked.reranker_rank,
                )
            )
        else:
            n_text += 1
            label = f"[T{n_text}]"
            views.append(
                EvidenceView(
                    label=label,
                    evidence_type="text",
                    document_id=fused.document_id,
                    source=source,
                    page=fused.page,
                    evidence_id=evidence_id,
                    text=str(payload.get(PAYLOAD_CHUNK_TEXT) or ""),
                    chunk_id=fused.chunk_id,
                    modality="text",
                    modality_sources=fused.modality_sources,
                    rrf_rank=reranked.original_rrf_rank,
                    reranker_rank=reranked.reranker_rank,
                )
            )
    return views


def _figure_description(payload: dict[str, Any]) -> str | None:
    """Best available production-derived description for a figure.

    Same precedence the reranker uses (VLM caption, then VLM description, then
    the nearby-text fallback), so what the generator reads about a figure is
    consistent with what the reranker scored it on.
    """
    for key in (PAYLOAD_FIGURE_CAPTION, PAYLOAD_FIGURE_DESCRIPTION, PAYLOAD_FIGURE_NEARBY_TEXT):
        value = payload.get(key)
        if value and str(value).strip():
            return str(value).strip()
    return None


def verify_answer_citations(answer: str, views: list[EvidenceView]) -> CitationVerification:
    """Resolve the labels an answer references against the evidence retrieved.

    Only labels the application assigned can resolve. Anything else the model
    emitted in label form is reported as unknown rather than being parsed into
    a citation, which is what stops an invented page or figure id from reaching
    the response.
    """
    import re

    known = {view.label for view in views}
    found = {f"[{kind}{number}]" for kind, number in re.findall(r"\[([TF])(\d+)\]", answer)}
    return CitationVerification(
        referenced=sorted(found & known),
        unknown=sorted(found - known),
    )


class CanonicalMultimodalRAG:
    """Production multimodal RAG over the PR4/PR5 canonical retrieval stack.

    Text retrieval is required; caption and CLIP retrieval and the reranker are
    optional and degrade independently (see ``retrieve_multimodal``).

    Whether figure *pixels* reach the model depends on the configured generator.
    Images are attached only when the generator accepts them and the figure has
    a usable image asset; otherwise the figure is represented to the model by
    its textual description alone. A filesystem path is never presented to the
    model as if it were visual content.
    """

    def __init__(
        self,
        text_store: Any,
        vlm: Any,
        caption_store: Any | None = None,
        image_store: Any | None = None,
        reranker: Any | None = None,
        top_k: int = 5,
        attach_images: bool = True,
    ) -> None:
        self._text_store = text_store
        self._vlm = vlm
        self._caption_store = caption_store
        self._image_store = image_store
        self._reranker = reranker
        self._top_k = top_k
        self._attach_images = attach_images

    def ask(self, question: str) -> MultimodalAnswer:
        """Retrieve → fuse → rerank → generate → verified citations."""
        return self._run(question)[0]

    def ask_with_diagnostics(
        self,
        question: str,
    ) -> tuple[MultimodalAnswer, RetrievalDiagnostics, CitationVerification]:
        """``ask`` plus the retrieval diagnostics and citation verification.

        Shares one execution with ``ask`` rather than repeating it, so the
        diagnostics returned here describe the very run that produced the
        answer.
        """
        return self._run(question)

    def _run(
        self,
        question: str,
    ) -> tuple[MultimodalAnswer, RetrievalDiagnostics, CitationVerification]:
        """Execute the full pipeline exactly once."""
        t0 = time.perf_counter()

        candidates, diagnostics = retrieve_multimodal(
            question,
            text_store=self._text_store,
            caption_store=self._caption_store,
            image_store=self._image_store,
            reranker=self._reranker,
            top_k=self._top_k,
        )

        views = build_evidence_views(candidates)
        text_views = [v for v in views if v.evidence_type == "text"]
        figure_views = [v for v in views if v.evidence_type == "figure"]

        prompt = load_prompt(
            "canonical_multimodal_rag",
            question=question,
            text_evidence=text_views,
            figure_evidence=figure_views,
        )

        images = self._collect_images(figure_views) if self._attach_images else []

        t_gen = time.perf_counter()
        try:
            answer = self._generate(prompt, images)
            mode = "multimodal" if images else "text_only"
        except LLMError:
            # The configured generator could not accept images (or was
            # unavailable in vision mode). Retry with identical evidence and no
            # attachments so the figure's textual description still grounds it.
            answer = self._generate(prompt, [])
            mode = "text_only"
        latency_generation = time.perf_counter() - t_gen

        verification = verify_answer_citations(answer, views)

        with trace_span(
            "mrta.canonical_rag.ask",
            {
                "generation.text_evidence_count": len(text_views),
                "generation.figure_evidence_count": len(figure_views),
                "generation.images_attached": len(images),
                "generation.retrieval_mode": mode,
                "generation.citations_referenced": len(verification.referenced),
                "generation.citations_unknown": len(verification.unknown),
                "retrieval.reranker_used": diagnostics.reranker_used,
                "retrieval.degraded_streams": sorted(diagnostics.degraded_streams),
                "latency.generation": round(latency_generation, 4),
            },
        ):
            pass

        result = MultimodalAnswer(
            answer=answer,
            text_citations=[v.to_citation() for v in text_views],
            visual_citations=[v.to_citation() for v in figure_views],
            retrieval_mode=mode,  # type: ignore[arg-type]
            latency_s=time.perf_counter() - t0,
        )
        return result, diagnostics, verification

    def _generate(self, prompt: str, images: list[PILImage]) -> str:
        """Call the configured generator, adapting to its supported interface."""
        generate = getattr(self._vlm, "generate", None)
        if generate is not None:
            return str(generate(prompt, images))
        # A text-only LLMClient: it has no image parameter at all, so figure
        # evidence reaches it purely as text.
        return str(self._vlm.chat([{"role": "user", "content": prompt}]))

    def _collect_images(self, figure_views: list[EvidenceView]) -> list[PILImage]:
        """Load image assets for figures that have a verified, existing path.

        A figure whose asset is missing keeps its textual evidence and simply
        contributes no image, rather than failing the query.
        """
        images: list[PILImage] = []
        for view in figure_views:
            if not view.image_path:
                continue
            try:
                from PIL import Image

                images.append(Image.open(view.image_path))
            except Exception:  # noqa: BLE001 — a missing asset must not fail the query
                continue
        return images
