"""Unit tests for LLMClient, rag_query, load_prompt, and StructuredLogger."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import mrta.observability.logging as log_module
from mrta.core.llm import LLMClient
from mrta.core.rag_pipeline import rag_query
from mrta.core.schemas import Chunk
from mrta.observability.logging import StructuredLogger
from mrta.prompts import load_prompt

FAKE_CHUNKS = [
    Chunk(
        chunk_id="doc_p1_c0",
        doc_id="doc",
        source="test.pdf",
        page=1,
        text="Attention is all you need.",
    ),
    Chunk(
        chunk_id="doc_p2_c0",
        doc_id="doc",
        source="test.pdf",
        page=2,
        text="Multi-head attention is powerful.",
    ),
]

MOCK_ANSWER = "According to [page 1], attention is the key mechanism."


class TestLLMClient:
    def test_chat_returns_mocked_text(self) -> None:
        with patch("mrta.core.llm.ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": MOCK_ANSWER}}
            client = LLMClient()
            result = client.chat([{"role": "user", "content": "What is attention?"}])
        assert result == MOCK_ANSWER

    def test_chat_returns_str(self) -> None:
        with patch("mrta.core.llm.ollama.chat") as mock_chat:
            mock_chat.return_value = {"message": {"content": "plain text"}}
            client = LLMClient()
            result = client.chat([{"role": "user", "content": "Q"}])
        assert isinstance(result, str)


class TestRagQuery:
    @pytest.fixture
    def mock_store(self) -> MagicMock:
        store = MagicMock()
        store.search_with_scores.return_value = [(c, 0.9) for c in FAKE_CHUNKS]
        return store

    @pytest.fixture
    def mock_llm(self) -> MagicMock:
        llm = MagicMock()
        llm.chat.return_value = MOCK_ANSWER
        return llm

    def test_returns_expected_keys(self, mock_store: MagicMock, mock_llm: MagicMock) -> None:
        result = rag_query("test?", mock_store, mock_llm)
        assert {"answer", "sources", "latency_s"} <= result.keys()

    def test_answer_is_str(self, mock_store: MagicMock, mock_llm: MagicMock) -> None:
        result = rag_query("test?", mock_store, mock_llm)
        assert isinstance(result["answer"], str)

    def test_sources_are_chunk_instances(self, mock_store: MagicMock, mock_llm: MagicMock) -> None:
        result = rag_query("test?", mock_store, mock_llm)
        for src in result["sources"]:
            assert isinstance(src, Chunk)

    def test_top_k_1_returns_one_source(self, mock_llm: MagicMock) -> None:
        store = MagicMock()
        store.search_with_scores.return_value = [(FAKE_CHUNKS[0], 0.9)]
        result = rag_query("test?", store, mock_llm, top_k=1)
        assert len(result["sources"]) == 1

    def test_latency_is_non_negative_float(
        self, mock_store: MagicMock, mock_llm: MagicMock
    ) -> None:
        result = rag_query("test?", mock_store, mock_llm)
        assert isinstance(result["latency_s"], float)
        assert result["latency_s"] >= 0


class TestLoadPrompt:
    def test_returns_nonempty_string(self) -> None:
        result = load_prompt("rag", chunks=[], question="test question")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_question(self) -> None:
        result = load_prompt("rag", chunks=[], question="test question")
        assert "test question" in result

    def test_renders_chunk_content(self) -> None:
        result = load_prompt("rag", chunks=FAKE_CHUNKS, question="Q")
        assert "Attention is all you need." in result


class TestStructuredLogger:
    def test_appends_one_line(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        log_file = tmp_path / "runs.jsonl"
        monkeypatch.setattr(log_module.settings, "log_file", log_file)
        logger = StructuredLogger()
        logger.log_run("test?", MOCK_ANSWER, FAKE_CHUNKS[:1], 1.23)
        lines = log_file.read_text().splitlines()
        assert len(lines) == 1

    def test_json_contains_question_and_answer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log_file = tmp_path / "runs.jsonl"
        monkeypatch.setattr(log_module.settings, "log_file", log_file)
        logger = StructuredLogger()
        logger.log_run("my question", "my answer", FAKE_CHUNKS[:1], 0.5)
        data = json.loads(log_file.read_text().splitlines()[0])
        assert data["question"] == "my question"
        assert data["answer"] == "my answer"
