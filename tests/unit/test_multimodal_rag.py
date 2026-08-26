"""Unit tests for MultimodalRAG.

All stores and the VLM are mocked — no Ollama, no CLIP, no FAISS in CI.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock

from PIL import Image

from mrta.core.exceptions import LLMError
from mrta.core.schemas import EvidenceRecord, MultimodalAnswer
from mrta.generation.multimodal_rag import MultimodalRAG

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text_record(eid: str, page: int = 1) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=eid,
        doc_id="doc1",
        source="paper.pdf",
        page=page,
        modality="text",
        text=f"Text content for {eid}.",
    )


def _make_image_record(
    eid: str, page: int = 2, with_bytes: bool = True, figure_index: int = 1
) -> EvidenceRecord:
    image_bytes: bytes | None = None
    if with_bytes:
        buf = io.BytesIO()
        Image.new("RGB", (10, 10), color=(128, 0, 0)).save(buf, format="PNG")
        image_bytes = buf.getvalue()
    return EvidenceRecord(
        evidence_id=eid,
        doc_id="doc1",
        source="paper.pdf",
        page=page,
        modality="image",
        figure_index=figure_index,
        image_bytes=image_bytes,
        caption="A diagram.",
    )


def _make_retriever(records: list[EvidenceRecord]) -> MagicMock:
    retriever = MagicMock()
    retriever.retrieve.return_value = records
    return retriever


def _make_vlm(answer: str = "The answer is [T1].") -> MagicMock:
    vlm = MagicMock()
    vlm.generate.return_value = answer
    return vlm


# ---------------------------------------------------------------------------
# TestMultimodalRAGBasic
# ---------------------------------------------------------------------------


class TestMultimodalRAGBasic:
    def test_ask_returns_multimodal_answer(self) -> None:
        retriever = _make_retriever([_make_text_record("t1"), _make_image_record("v1")])
        rag = MultimodalRAG(retriever=retriever, vlm=_make_vlm())
        assert isinstance(rag.ask("What is shown?"), MultimodalAnswer)

    def test_ask_answer_string_populated(self) -> None:
        retriever = _make_retriever([_make_text_record("t1")])
        rag = MultimodalRAG(retriever=retriever, vlm=_make_vlm("The answer is here."))
        assert rag.ask("Q?").answer == "The answer is here."

    def test_latency_s_populated(self) -> None:
        retriever = _make_retriever([_make_text_record("t1")])
        rag = MultimodalRAG(retriever=retriever, vlm=_make_vlm())
        assert rag.ask("Q?").latency_s > 0

    def test_retrieval_mode_multimodal_on_success(self) -> None:
        retriever = _make_retriever([_make_text_record("t1"), _make_image_record("v1")])
        rag = MultimodalRAG(retriever=retriever, vlm=_make_vlm())
        assert rag.ask("Q?").retrieval_mode == "multimodal"

    def test_empty_retrieval_still_calls_vlm(self) -> None:
        retriever = _make_retriever([])
        vlm = _make_vlm("No evidence found.")
        rag = MultimodalRAG(retriever=retriever, vlm=vlm)
        result = rag.ask("Q?")
        vlm.generate.assert_called_once()
        assert isinstance(result, MultimodalAnswer)

    def test_retriever_called_with_configured_k(self) -> None:
        retriever = _make_retriever([])
        rag = MultimodalRAG(
            retriever=retriever, vlm=_make_vlm(), text_top_k=3, visual_top_k=4, fusion_top_k=6
        )
        rag.ask("Q?")
        retriever.retrieve.assert_called_once_with("Q?", k_text=3, k_visual=4, k_final=6)


# ---------------------------------------------------------------------------
# TestCitationLabels
# ---------------------------------------------------------------------------


class TestCitationLabels:
    def test_text_citation_labels(self) -> None:
        retriever = _make_retriever([_make_text_record("t1"), _make_text_record("t2")])
        result = MultimodalRAG(retriever=retriever, vlm=_make_vlm()).ask("Q?")
        assert [c.label for c in result.text_citations] == ["[T1]", "[T2]"]

    def test_visual_citation_labels(self) -> None:
        retriever = _make_retriever([_make_image_record("v1"), _make_image_record("v2", page=3)])
        result = MultimodalRAG(retriever=retriever, vlm=_make_vlm()).ask("Q?")
        assert [c.label for c in result.visual_citations] == ["[V1]", "[V2]"]

    def test_text_citation_fields(self) -> None:
        rec = _make_text_record("t1", page=7)
        result = MultimodalRAG(retriever=_make_retriever([rec]), vlm=_make_vlm()).ask("Q?")
        cit = result.text_citations[0]
        assert cit.evidence_id == "t1"
        assert cit.source == "paper.pdf"
        assert cit.page == 7
        assert cit.modality == "text"

    def test_visual_citation_figure_index(self) -> None:
        rec = _make_image_record("v1", figure_index=3)
        result = MultimodalRAG(retriever=_make_retriever([rec]), vlm=_make_vlm()).ask("Q?")
        assert result.visual_citations[0].figure_index == 3

    def test_text_citations_modality(self) -> None:
        retriever = _make_retriever([_make_text_record("t1"), _make_text_record("t2")])
        result = MultimodalRAG(retriever=retriever, vlm=_make_vlm()).ask("Q?")
        assert all(c.modality == "text" for c in result.text_citations)

    def test_visual_citations_modality(self) -> None:
        retriever = _make_retriever([_make_image_record("v1"), _make_image_record("v2")])
        result = MultimodalRAG(retriever=retriever, vlm=_make_vlm()).ask("Q?")
        assert all(c.modality in ("image", "page") for c in result.visual_citations)

    def test_text_only_evidence_no_visual_citations(self) -> None:
        retriever = _make_retriever([_make_text_record("t1"), _make_text_record("t2")])
        result = MultimodalRAG(retriever=retriever, vlm=_make_vlm()).ask("Q?")
        assert result.visual_citations == []
        assert len(result.text_citations) == 2


# ---------------------------------------------------------------------------
# TestImagePassing
# ---------------------------------------------------------------------------


class TestImagePassing:
    def test_images_with_bytes_passed_to_vlm(self) -> None:
        retriever = _make_retriever([_make_image_record("v1", with_bytes=True)])
        vlm = _make_vlm()
        MultimodalRAG(retriever=retriever, vlm=vlm).ask("Q?")
        _, images = vlm.generate.call_args.args
        assert len(images) == 1

    def test_images_without_bytes_excluded_from_vlm(self) -> None:
        retriever = _make_retriever([_make_image_record("v1", with_bytes=False)])
        vlm = _make_vlm()
        MultimodalRAG(retriever=retriever, vlm=vlm).ask("Q?")
        _, images = vlm.generate.call_args.args
        assert images == []

    def test_multiple_images_passed_to_vlm(self) -> None:
        records = [
            _make_image_record("v1"),
            _make_image_record("v2", page=3),
            _make_image_record("v3", page=5),
        ]
        vlm = _make_vlm()
        MultimodalRAG(retriever=_make_retriever(records), vlm=vlm).ask("Q?")
        _, images = vlm.generate.call_args.args
        assert len(images) == 3

    def test_vlm_receives_pil_images_not_strings(self) -> None:
        retriever = _make_retriever([_make_image_record("v1", with_bytes=True)])
        vlm = _make_vlm()
        MultimodalRAG(retriever=retriever, vlm=vlm).ask("Q?")
        _, images = vlm.generate.call_args.args
        assert all(isinstance(img, Image.Image) for img in images)


# ---------------------------------------------------------------------------
# TestFallback
# ---------------------------------------------------------------------------


class TestFallback:
    def test_vlm_error_falls_back_to_text_only(self) -> None:
        retriever = _make_retriever([_make_text_record("t1"), _make_image_record("v1")])
        vlm = MagicMock()
        vlm.generate.side_effect = [LLMError("model not found"), "Fallback answer."]
        result = MultimodalRAG(retriever=retriever, vlm=vlm).ask("Q?")
        assert result.answer == "Fallback answer."
        assert result.retrieval_mode == "text_only"

    def test_fallback_does_not_pass_images(self) -> None:
        retriever = _make_retriever([_make_text_record("t1"), _make_image_record("v1")])
        vlm = MagicMock()
        vlm.generate.side_effect = [LLMError("unavailable"), "Text answer."]
        MultimodalRAG(retriever=retriever, vlm=vlm).ask("Q?")
        fallback_images = vlm.generate.call_args_list[1].args[1]
        assert fallback_images == []

    def test_text_citations_preserved_in_fallback(self) -> None:
        retriever = _make_retriever([_make_text_record("t1"), _make_image_record("v1")])
        vlm = MagicMock()
        vlm.generate.side_effect = [LLMError("unavailable"), "Answer."]
        result = MultimodalRAG(retriever=retriever, vlm=vlm).ask("Q?")
        assert len(result.text_citations) == 1
        assert result.text_citations[0].label == "[T1]"
