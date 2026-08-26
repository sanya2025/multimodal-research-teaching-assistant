"""mrta.generation.multimodal_rag — Grounded multimodal RAG generation.

Pipeline:
    question
    → MultimodalRetriever.retrieve()        (text + visual, RRF-fused)
    → split into text / visual evidence
    → render multimodal_rag.j2 prompt
    → collect original PIL images from visual records with image_bytes
    → VLMClient.generate(prompt, images)
    → MultimodalAnswer with [T#] / [V#] structured citations

Fallback: if VLMClient.generate() raises LLMError (model unavailable or
does not support vision), the method retries with the same evidence but no
images attached and sets retrieval_mode="text_only" in the returned answer.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from mrta.core.exceptions import LLMError
from mrta.core.schemas import EvidenceRecord, MultimodalAnswer, MultimodalCitation
from mrta.observability.tracing import trace_span
from mrta.prompts import load_prompt

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    from mrta.multimodal.vlm_client import VLMClient
    from mrta.retrieval.multimodal_retriever import MultimodalRetriever


class MultimodalRAG:
    """Grounded multimodal RAG: retrieve text + visual evidence, generate via VLM.

    At minimum, a MultimodalRetriever (text-only is fine) and VLMClient are required.
    The retriever degrades automatically when no visual store is configured.

    ``teaching_mode`` selects a mode-specific Jinja2 template instead of the default
    ``multimodal_rag.j2``. Valid values: "explain", "socratic", "quiz", "compare",
    "visual_evidence". ``None`` (default) uses the standard grounded-answer template.
    """

    VALID_TEACHING_MODES: frozenset[str] = frozenset(
        {"explain", "socratic", "quiz", "compare", "visual_evidence"}
    )

    def __init__(
        self,
        retriever: MultimodalRetriever,
        vlm: VLMClient,
        text_top_k: int = 5,
        visual_top_k: int = 5,
        fusion_top_k: int = 8,
        teaching_mode: str | None = None,
    ) -> None:
        if teaching_mode is not None and teaching_mode not in self.VALID_TEACHING_MODES:
            raise ValueError(
                f"teaching_mode must be one of {sorted(self.VALID_TEACHING_MODES)!r},"
                f" got {teaching_mode!r}"
            )
        self._retriever = retriever
        self._vlm = vlm
        self._text_top_k = text_top_k
        self._visual_top_k = visual_top_k
        self._fusion_top_k = fusion_top_k
        self._teaching_mode = teaching_mode

    def ask(self, question: str) -> MultimodalAnswer:
        """Full multimodal RAG: retrieve → fuse → prompt + images → VLM → grounded answer.

        Fallback: if VLMClient raises LLMError (e.g. model unavailable or
        does not support vision input), retries with the same text evidence
        but no images and sets retrieval_mode="text_only" in the result.
        """
        t0 = time.time()

        evidence = self._retriever.retrieve(
            question,
            k_text=self._text_top_k,
            k_visual=self._visual_top_k,
            k_final=self._fusion_top_k,
        )

        text_ev, visual_ev = self._split_evidence(evidence)
        text_cits, visual_cits = self._make_citations(text_ev, visual_ev)

        vlm_model = getattr(self._vlm, "_model", "unknown")

        try:
            prompt, images = self._build_prompt_and_images(question, text_ev, visual_ev)
            t_vlm = time.time()
            answer = self._vlm.generate(prompt, images)
            latency_vlm = time.time() - t_vlm
            mode: str = "multimodal"
        except LLMError:
            # Retry without images — useful when the model does not support vision
            fallback_prompt, _ = self._build_prompt_and_images(question, text_ev, [])
            t_vlm = time.time()
            answer = self._vlm.generate(fallback_prompt, [])
            latency_vlm = time.time() - t_vlm
            mode = "text_only"

        latency_s = time.time() - t0

        with trace_span(
            "mrta.multimodal_rag.ask",
            {
                "generation.text_evidence_count": len(text_ev),
                "generation.visual_evidence_count": len(visual_ev),
                "generation.vision_model": str(vlm_model),
                "generation.teaching_mode": self._teaching_mode or "none",
                "generation.retrieval_mode": mode,
                "latency.vlm": round(latency_vlm, 4),
            },
        ):
            pass

        return MultimodalAnswer(
            answer=answer,
            text_citations=text_cits,
            visual_citations=visual_cits,
            retrieval_mode=mode,  # type: ignore[arg-type]
            latency_s=latency_s,
        )

    def _split_evidence(
        self,
        evidence: list[EvidenceRecord],
    ) -> tuple[list[EvidenceRecord], list[EvidenceRecord]]:
        """Separate fused evidence into text and visual lists, preserving RRF order."""
        text_ev = [e for e in evidence if e.modality == "text"]
        visual_ev = [e for e in evidence if e.modality in ("image", "page")]
        return text_ev, visual_ev

    def _build_prompt_and_images(
        self,
        question: str,
        text_ev: list[EvidenceRecord],
        visual_ev: list[EvidenceRecord],
    ) -> tuple[str, list[PILImage]]:
        """Render multimodal_rag.j2 and collect PIL images from records with bytes.

        Records in visual_ev that have no image_bytes (e.g. vector-graphic pages)
        appear in the prompt text but are not attached as images to the VLM call.
        """
        template = f"teaching_{self._teaching_mode}" if self._teaching_mode else "multimodal_rag"
        prompt = load_prompt(
            template,
            question=question,
            text_evidence=text_ev,
            visual_evidence=visual_ev,
        )
        images = [ev.to_pil() for ev in visual_ev if ev.image_bytes is not None]
        return prompt, images

    def _make_citations(
        self,
        text_ev: list[EvidenceRecord],
        visual_ev: list[EvidenceRecord],
    ) -> tuple[list[MultimodalCitation], list[MultimodalCitation]]:
        """Assign [T1]…[Tn] labels to text evidence and [V1]…[Vn] to visual."""
        text_cits = [
            MultimodalCitation(
                label=f"[T{i + 1}]",
                evidence_id=ev.evidence_id,
                modality=ev.modality,
                source=ev.source,
                page=ev.page,
                figure_index=ev.figure_index,
            )
            for i, ev in enumerate(text_ev)
        ]
        visual_cits = [
            MultimodalCitation(
                label=f"[V{i + 1}]",
                evidence_id=ev.evidence_id,
                modality=ev.modality,
                source=ev.source,
                page=ev.page,
                figure_index=ev.figure_index,
            )
            for i, ev in enumerate(visual_ev)
        ]
        return text_cits, visual_cits
