"""Unit tests for MultimodalRAG teaching modes.

All stores and VLM are mocked — no Ollama, no CLIP, no FAISS in CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mrta.core.schemas import EvidenceRecord, MultimodalAnswer
from mrta.generation.multimodal_rag import MultimodalRAG

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text_record(eid: str = "t1") -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=eid,
        doc_id="doc1",
        source="paper.pdf",
        page=1,
        modality="text",
        text="Some content.",
    )


def _make_retriever(records: list[EvidenceRecord] | None = None) -> MagicMock:
    retriever = MagicMock()
    retriever.retrieve.return_value = records or [_make_text_record()]
    return retriever


def _make_vlm(answer: str = "The answer.") -> MagicMock:
    vlm = MagicMock()
    vlm.generate.return_value = answer
    return vlm


# ---------------------------------------------------------------------------
# TestTeachingModeDispatch
# ---------------------------------------------------------------------------


class TestTeachingModeDispatch:
    """Teaching mode selects the correct Jinja2 template."""

    @pytest.mark.parametrize(
        "mode, expected_template",
        [
            (None, "multimodal_rag"),
            ("explain", "teaching_explain"),
            ("socratic", "teaching_socratic"),
            ("quiz", "teaching_quiz"),
            ("compare", "teaching_compare"),
            ("visual_evidence", "teaching_visual_evidence"),
        ],
    )
    def test_correct_template_loaded(self, mode: str | None, expected_template: str) -> None:
        rag = MultimodalRAG(retriever=_make_retriever(), vlm=_make_vlm(), teaching_mode=mode)
        with patch("mrta.generation.multimodal_rag.load_prompt") as mock_load:
            mock_load.return_value = "prompt"
            rag.ask("What is attention?")
        called_template = mock_load.call_args[0][0]
        assert called_template == expected_template

    def test_no_teaching_mode_uses_default_template(self) -> None:
        rag = MultimodalRAG(retriever=_make_retriever(), vlm=_make_vlm())
        with patch("mrta.generation.multimodal_rag.load_prompt") as mock_load:
            mock_load.return_value = "prompt"
            rag.ask("Q?")
        assert mock_load.call_args[0][0] == "multimodal_rag"

    def test_invalid_teaching_mode_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="teaching_mode must be one of"):
            MultimodalRAG(retriever=_make_retriever(), vlm=_make_vlm(), teaching_mode="lecture")

    def test_all_valid_modes_accepted(self) -> None:
        for mode in MultimodalRAG.VALID_TEACHING_MODES:
            rag = MultimodalRAG(retriever=_make_retriever(), vlm=_make_vlm(), teaching_mode=mode)
            assert rag._teaching_mode == mode


# ---------------------------------------------------------------------------
# TestTeachingModeOutput
# ---------------------------------------------------------------------------


class TestTeachingModeOutput:
    """All teaching modes return a valid MultimodalAnswer."""

    @pytest.mark.parametrize(
        "mode", [None, "explain", "socratic", "quiz", "compare", "visual_evidence"]
    )
    def test_returns_multimodal_answer(self, mode: str | None) -> None:
        rag = MultimodalRAG(retriever=_make_retriever(), vlm=_make_vlm(), teaching_mode=mode)
        result = rag.ask("What is attention?")
        assert isinstance(result, MultimodalAnswer)

    @pytest.mark.parametrize("mode", ["explain", "socratic", "quiz", "compare", "visual_evidence"])
    def test_answer_populated_for_all_modes(self, mode: str) -> None:
        rag = MultimodalRAG(
            retriever=_make_retriever(), vlm=_make_vlm(f"{mode} answer"), teaching_mode=mode
        )
        assert rag.ask("Q?").answer == f"{mode} answer"

    @pytest.mark.parametrize("mode", ["explain", "socratic", "quiz", "compare", "visual_evidence"])
    def test_retrieval_mode_multimodal_for_all_teaching_modes(self, mode: str) -> None:
        rag = MultimodalRAG(retriever=_make_retriever(), vlm=_make_vlm(), teaching_mode=mode)
        assert rag.ask("Q?").retrieval_mode == "multimodal"

    def test_teaching_mode_stored_on_instance(self) -> None:
        rag = MultimodalRAG(retriever=_make_retriever(), vlm=_make_vlm(), teaching_mode="quiz")
        assert rag._teaching_mode == "quiz"

    def test_none_teaching_mode_stored_on_instance(self) -> None:
        rag = MultimodalRAG(retriever=_make_retriever(), vlm=_make_vlm())
        assert rag._teaching_mode is None


# ---------------------------------------------------------------------------
# TestTeachingModePromptVariables
# ---------------------------------------------------------------------------


class TestTeachingModePromptVariables:
    """load_prompt receives the correct kwargs for all teaching modes."""

    def test_question_passed_to_template(self) -> None:
        rag = MultimodalRAG(retriever=_make_retriever(), vlm=_make_vlm(), teaching_mode="explain")
        with patch("mrta.generation.multimodal_rag.load_prompt") as mock_load:
            mock_load.return_value = "prompt"
            rag.ask("Explain attention.")
        kwargs = mock_load.call_args[1]
        assert kwargs["question"] == "Explain attention."

    def test_text_evidence_passed_to_template(self) -> None:
        rag = MultimodalRAG(retriever=_make_retriever(), vlm=_make_vlm(), teaching_mode="socratic")
        with patch("mrta.generation.multimodal_rag.load_prompt") as mock_load:
            mock_load.return_value = "prompt"
            rag.ask("Q?")
        kwargs = mock_load.call_args[1]
        assert "text_evidence" in kwargs

    def test_visual_evidence_passed_to_template(self) -> None:
        rag = MultimodalRAG(retriever=_make_retriever(), vlm=_make_vlm(), teaching_mode="compare")
        with patch("mrta.generation.multimodal_rag.load_prompt") as mock_load:
            mock_load.return_value = "prompt"
            rag.ask("Q?")
        kwargs = mock_load.call_args[1]
        assert "visual_evidence" in kwargs
