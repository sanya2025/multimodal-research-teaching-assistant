"""Tests for mrta.prompts — load_prompt and MODES."""

from __future__ import annotations

import jinja2
import pytest

from mrta.prompts import MODES, load_prompt

FAKE_CHUNKS = [{"source": "test.pdf", "page": 1, "text": "Some context text."}]


class TestLoadPrompt:
    def test_rag_contains_question(self) -> None:
        result = load_prompt("rag", question="Q", chunks=[])
        assert result.strip()
        assert "Q" in result

    def test_beginner_contains_question(self) -> None:
        result = load_prompt("beginner", question="test question", chunks=[])
        assert result.strip()
        assert "test question" in result

    def test_expert_returns_nonempty(self) -> None:
        result = load_prompt("expert", question="Q", chunks=[])
        assert result.strip()

    def test_quiz_contains_quiz_marker(self) -> None:
        result = load_prompt("quiz", question="Q", chunks=[])
        assert "QUIZ" in result

    def test_interview_returns_nonempty(self) -> None:
        result = load_prompt("interview", question="Q", chunks=[])
        assert result.strip()

    def test_lecture_notes_returns_nonempty(self) -> None:
        result = load_prompt("lecture_notes", question="Q", chunks=[])
        assert result.strip()

    def test_explain_no_kwargs(self) -> None:
        result = load_prompt("explain")
        assert result.strip()

    def test_explain_level_injected(self) -> None:
        result = load_prompt("explain", level="high school student")
        assert "high school student" in result

    def test_explain_question_injected(self) -> None:
        result = load_prompt("explain", question="What does the x-axis represent?")
        assert "What does the x-axis represent?" in result

    def test_unknown_template_raises(self) -> None:
        with pytest.raises(jinja2.exceptions.TemplateNotFound):
            load_prompt("nonexistent")

    def test_chunks_rendered_in_rag(self) -> None:
        result = load_prompt("rag", question="Q", chunks=FAKE_CHUNKS)
        assert "test.pdf" in result
        assert "Some context text." in result


class TestModes:
    def test_modes_is_dict(self) -> None:
        assert isinstance(MODES, dict)

    def test_modes_has_required_keys(self) -> None:
        required = {"beginner", "expert", "quiz", "lecture_notes", "interview", "explain"}
        assert required.issubset(MODES.keys())

    def test_all_rag_modes_loadable(self) -> None:
        rag_modes = {k for k in MODES if k not in ("explain", "default")}
        for mode in rag_modes:
            result = load_prompt(MODES[mode], question="Q", chunks=[])
            assert result.strip(), f"Mode '{mode}' rendered empty string"

    def test_explain_mode_loadable(self) -> None:
        result = load_prompt(MODES["explain"])
        assert result.strip()
