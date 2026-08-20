"""mrta.multimodal.visual_analyzer — structured VLM-based description of figures.

Produces a VisualDescription — a typed, serializable semantic breakdown of a
figure or page image — whose text representation is suitable for embedding and
retrieval. This is the bridge between raw images and the text retrieval pipeline.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from PIL import Image

    from mrta.core.schemas import EvidenceRecord, FigureRecord


_VISUAL_TYPES = (
    "architecture_diagram, graph, table, equation, photo, "
    "flowchart, heatmap, scatter_plot, bar_chart, line_chart, illustration, other"
)

_ANALYSIS_PROMPT = f"""\
You are analyzing a figure or diagram from a scientific or technical document.

Respond with a valid JSON object. Use null for any field you cannot determine.

{{
  "visual_type": "one of: {_VISUAL_TYPES}",
  "short_caption": "one-sentence description (20 words or fewer)",
  "detailed_description": "2-4 sentence description of what is shown",
  "objects": ["list of key objects, components, or elements visible"],
  "labels": ["axis labels, legend entries, or text labels in the figure"],
  "relationships": ["relationships or connections described or shown"],
  "trends": ["trends, patterns, or comparisons visible"],
  "main_conclusion": "the main point or takeaway (20 words or fewer, or null)"
}}

Respond with the JSON object only. No markdown fences, no explanation."""


class VisualDescription(BaseModel):
    """Structured semantic description of a single figure or page image."""

    visual_type: str | None = None
    short_caption: str | None = None
    detailed_description: str | None = None
    objects: list[str] = []
    labels: list[str] = []
    relationships: list[str] = []
    trends: list[str] = []
    main_conclusion: str | None = None

    def to_retrieval_text(
        self,
        source: str = "",
        page: int | None = None,
        figure_index: int | None = None,
        nearby_text: str | None = None,
    ) -> str:
        """Deterministic text representation for embedding and retrieval indexing.

        The format is intentionally verbose so that a text embedding model can
        capture semantic content that a one-sentence caption would miss.
        """
        lines: list[str] = ["[VISUAL EVIDENCE]"]
        if source:
            lines.append(f"Document: {source}")
        if page is not None:
            lines.append(f"Page: {page}")
        if figure_index is not None:
            lines.append(f"Figure: {figure_index}")
        if self.visual_type:
            lines.append(f"Type: {self.visual_type}")
        lines.append("")

        if self.short_caption:
            lines.append(f"Summary: {self.short_caption}")
        if self.detailed_description:
            lines.append(f"\nDetails:\n{self.detailed_description}")
        if self.objects:
            lines.append("\nComponents:\n" + "\n".join(f"- {o}" for o in self.objects))
        if self.labels:
            lines.append("\nLabels:\n" + "\n".join(f"- {lb}" for lb in self.labels))
        if self.relationships:
            lines.append("\nRelationships:\n" + "\n".join(f"- {r}" for r in self.relationships))
        if self.trends:
            lines.append("\nTrends:\n" + "\n".join(f"- {t}" for t in self.trends))
        if self.main_conclusion:
            lines.append(f"\nConclusion: {self.main_conclusion}")
        if nearby_text:
            lines.append(f"\nNearby text:\n{nearby_text}")

        return "\n".join(lines)


class VisualAnalyzer:
    """Analyzes figure images using a VLM and returns structured VisualDescription objects.

    Requires the [multimodal] optional dependency (open-clip-torch) and a running
    Ollama VLM (e.g. qwen2.5vl:latest). Falls back gracefully when the VLM is
    unavailable — returns an empty VisualDescription rather than raising.
    """

    def __init__(self, vlm: object | None = None) -> None:
        """
        Args:
            vlm: A VLMClient instance. If None, one is created from settings.
                 Accepts any object with a generate(prompt, images) -> str method,
                 so tests can inject a mock without importing VLMClient.
        """
        self._vlm = vlm

    def _get_vlm(self) -> object:
        if self._vlm is None:
            from mrta.multimodal.vlm_client import VLMClient

            self._vlm = VLMClient()
        return self._vlm

    def analyze(
        self,
        image: Image.Image,
        source: str = "",
        page: int | None = None,
        figure_index: int | None = None,
    ) -> VisualDescription:
        """Analyze a PIL image and return a structured VisualDescription.

        Returns an empty VisualDescription if the VLM is unavailable or returns
        unparseable output. Callers should check whether key fields are populated.
        """
        vlm = self._get_vlm()
        try:
            raw = vlm.generate(prompt=_ANALYSIS_PROMPT, images=[image])  # type: ignore[union-attr]
            data = json.loads(raw)
            return VisualDescription.model_validate(data)
        except Exception:
            return VisualDescription()

    def analyze_evidence(self, record: EvidenceRecord | FigureRecord) -> VisualDescription:
        """Analyze a FigureRecord or EvidenceRecord and mutate it with the description.

        Returns the VisualDescription and also sets caption, detailed_description,
        visual_type, and nearby_text on the record if it is an EvidenceRecord.
        """
        image = record.to_pil()

        # Resolve metadata depending on record type
        from mrta.core.schemas import EvidenceRecord as ER

        source = record.source
        page = record.page
        figure_index = getattr(record, "figure_index", None)

        desc = self.analyze(image, source=source, page=page, figure_index=figure_index)

        if isinstance(record, ER):
            record.caption = desc.short_caption
            record.detailed_description = desc.detailed_description
            record.visual_type = desc.visual_type

        return desc
