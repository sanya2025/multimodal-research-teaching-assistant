"""Tests for VisualDescription and VisualAnalyzer."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from PIL import Image

from mrta.multimodal.visual_analyzer import VisualAnalyzer, VisualDescription

_WHITE = Image.new("RGB", (4, 4), color=(255, 255, 255))


class TestVisualDescription:
    def test_default_fields_empty(self) -> None:
        desc = VisualDescription()
        assert desc.visual_type is None
        assert desc.short_caption is None
        assert desc.objects == []
        assert desc.labels == []

    def test_construction_with_fields(self) -> None:
        desc = VisualDescription(
            visual_type="architecture_diagram",
            short_caption="Transformer encoder-decoder.",
            objects=["encoder", "decoder"],
        )
        assert desc.visual_type == "architecture_diagram"
        assert "encoder" in desc.objects

    def test_to_retrieval_text_contains_header(self) -> None:
        desc = VisualDescription(short_caption="A graph.")
        text = desc.to_retrieval_text()
        assert "[VISUAL EVIDENCE]" in text

    def test_to_retrieval_text_includes_source_page_figure(self) -> None:
        desc = VisualDescription()
        text = desc.to_retrieval_text(source="paper.pdf", page=3, figure_index=2)
        assert "Document: paper.pdf" in text
        assert "Page: 3" in text
        assert "Figure: 2" in text

    def test_to_retrieval_text_includes_caption(self) -> None:
        desc = VisualDescription(short_caption="An attention heatmap.")
        text = desc.to_retrieval_text()
        assert "An attention heatmap." in text

    def test_to_retrieval_text_includes_objects(self) -> None:
        desc = VisualDescription(objects=["encoder block", "residual connection"])
        text = desc.to_retrieval_text()
        assert "encoder block" in text
        assert "residual connection" in text

    def test_to_retrieval_text_includes_nearby_text(self) -> None:
        desc = VisualDescription()
        text = desc.to_retrieval_text(nearby_text="See Figure 3 for details.")
        assert "See Figure 3 for details." in text

    def test_to_retrieval_text_omits_empty_fields(self) -> None:
        desc = VisualDescription(short_caption="Only caption.")
        text = desc.to_retrieval_text()
        assert "Components:" not in text
        assert "Labels:" not in text

    def test_to_retrieval_text_is_deterministic(self) -> None:
        desc = VisualDescription(
            visual_type="graph",
            short_caption="BLEU score comparison.",
            objects=["model A", "model B"],
            main_conclusion="Model A outperforms Model B.",
        )
        assert desc.to_retrieval_text() == desc.to_retrieval_text()

    def test_round_trip_json(self) -> None:
        desc = VisualDescription(
            visual_type="bar_chart",
            short_caption="Results comparison.",
            labels=["x-axis", "y-axis"],
        )
        serialised = desc.model_dump_json()
        restored = VisualDescription.model_validate_json(serialised)
        assert restored.visual_type == desc.visual_type
        assert restored.labels == desc.labels


class TestVisualAnalyzer:
    def _make_vlm(self, response: str) -> MagicMock:
        vlm = MagicMock()
        vlm.generate.return_value = response
        return vlm

    def test_analyze_returns_visual_description(self) -> None:
        payload = json.dumps(
            {
                "visual_type": "architecture_diagram",
                "short_caption": "Transformer model.",
                "detailed_description": "The figure shows an encoder and decoder.",
                "objects": ["encoder", "decoder"],
                "labels": [],
                "relationships": ["encoder feeds decoder"],
                "trends": [],
                "main_conclusion": "Standard seq2seq architecture.",
            }
        )
        analyzer = VisualAnalyzer(vlm=self._make_vlm(payload))
        desc = analyzer.analyze(_WHITE)
        assert isinstance(desc, VisualDescription)
        assert desc.visual_type == "architecture_diagram"
        assert desc.short_caption == "Transformer model."
        assert "encoder" in desc.objects

    def test_analyze_passes_image_to_vlm(self) -> None:
        vlm = self._make_vlm(json.dumps({}))
        analyzer = VisualAnalyzer(vlm=vlm)
        analyzer.analyze(_WHITE)
        vlm.generate.assert_called_once()
        call_kwargs = vlm.generate.call_args
        images_arg = call_kwargs.kwargs.get("images") or call_kwargs.args[1]
        assert len(images_arg) == 1

    def test_analyze_returns_empty_on_invalid_json(self) -> None:
        analyzer = VisualAnalyzer(vlm=self._make_vlm("not valid json {{{"))
        desc = analyzer.analyze(_WHITE)
        assert isinstance(desc, VisualDescription)
        assert desc.visual_type is None

    def test_analyze_returns_empty_on_vlm_exception(self) -> None:
        vlm = MagicMock()
        vlm.generate.side_effect = RuntimeError("VLM unreachable")
        analyzer = VisualAnalyzer(vlm=vlm)
        desc = analyzer.analyze(_WHITE)
        assert isinstance(desc, VisualDescription)

    def test_analyze_partial_json_populates_available_fields(self) -> None:
        payload = json.dumps({"visual_type": "graph"})
        analyzer = VisualAnalyzer(vlm=self._make_vlm(payload))
        desc = analyzer.analyze(_WHITE)
        assert desc.visual_type == "graph"
        assert desc.short_caption is None
