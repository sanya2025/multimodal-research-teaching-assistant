"""Tests for mrta.core.exceptions — hierarchy and raise-site integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from mrta.core.exceptions import (
    EmbeddingError,
    EvaluationError,
    IngestionError,
    LLMError,
    MRTAError,
    RetrievalError,
)


class TestHierarchy:
    def test_base_is_exception(self) -> None:
        assert issubclass(MRTAError, Exception)

    def test_all_subclasses_are_mrta_error(self) -> None:
        for cls in (IngestionError, EmbeddingError, RetrievalError, LLMError, EvaluationError):
            assert issubclass(cls, MRTAError)


class TestRaiseSites:
    def test_pdf_loader_raises_ingestion_error(self) -> None:
        from mrta.ingestion.pdf_loader import load_pdf

        with patch("fitz.open", side_effect=RuntimeError("bad PDF")):
            with pytest.raises(IngestionError) as exc_info:
                load_pdf("fake.pdf")
        assert exc_info.value.__cause__ is not None

    def test_embedder_raises_embedding_error(self) -> None:
        from mrta.retrieval.embedder import Embedder

        embedder = Embedder.__new__(Embedder)
        embedder._model_name = "nomic-embed-text"
        embedder._use_st = False

        req = httpx.Request("POST", "http://localhost/api/embeddings")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404", request=req, response=httpx.Response(404, request=req)
        )
        with patch("httpx.post", return_value=mock_resp):
            with pytest.raises(EmbeddingError) as exc_info:
                embedder._embed_ollama(["test"])
        assert exc_info.value.__cause__ is not None

    def test_vector_store_raises_retrieval_error(self, tmp_path) -> None:
        from mrta.retrieval.vector_store import VectorStore

        mock_embedder = MagicMock()
        mock_embedder.dim = 384
        with (
            patch("faiss.IndexFlatIP"),
            patch("faiss.read_index", side_effect=RuntimeError("corrupt index")),
        ):
            with pytest.raises(RetrievalError) as exc_info:
                VectorStore.load(tmp_path, mock_embedder)
        assert exc_info.value.__cause__ is not None

    def test_llm_raises_llm_error(self) -> None:
        from mrta.core.llm import LLMClient

        llm = LLMClient.__new__(LLMClient)
        llm._provider = "ollama"
        llm._model = "llama3.2"
        with patch("mrta.core.llm.ollama.chat", side_effect=Exception("connection refused")):
            with pytest.raises(LLMError) as exc_info:
                llm.chat([{"role": "user", "content": "hello"}])
        assert exc_info.value.__cause__ is not None

    def test_vlm_client_raises_llm_error(self) -> None:
        from PIL import Image

        from mrta.multimodal.vlm_client import VLMClient

        vlm = VLMClient.__new__(VLMClient)
        vlm._model = "qwen2.5vl:7b"
        image = Image.new("RGB", (10, 10))
        with patch("mrta.multimodal.vlm_client.ollama.chat", side_effect=Exception("no GPU")):
            with pytest.raises(LLMError) as exc_info:
                vlm.caption(image)
        assert exc_info.value.__cause__ is not None


class TestExport:
    def test_importable_from_mrta(self) -> None:
        from mrta import (  # noqa: F401
            EmbeddingError,
            EvaluationError,
            IngestionError,
            LLMError,
            MRTAError,
            RetrievalError,
        )
