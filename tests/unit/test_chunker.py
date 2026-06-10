"""Unit tests for mrta.ingestion.chunker."""

from __future__ import annotations

from pathlib import Path

import pytest

from mrta.core.schemas import Chunk
from mrta.ingestion.chunker import chunk_pdf, fixed_chunks
from mrta.ingestion.pdf_loader import load_pdf

FIXTURE_PDF = Path(__file__).parents[1] / "fixtures" / "sample.pdf"


@pytest.fixture(scope="module")
def pdf():
    return load_pdf(FIXTURE_PDF)


class TestFixedChunks:
    def test_returns_chunks(self, pdf) -> None:
        chunks = fixed_chunks(pdf)
        assert len(chunks) > 0

    def test_all_are_chunk_instances(self, pdf) -> None:
        for c in fixed_chunks(pdf):
            assert isinstance(c, Chunk)

    def test_chunk_ids_unique(self, pdf) -> None:
        chunks = fixed_chunks(pdf)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_all_chunks_have_required_fields(self, pdf) -> None:
        for c in fixed_chunks(pdf):
            assert c.doc_id
            assert c.source
            assert c.page >= 1
            assert c.text.strip()

    def test_chunk_id_format(self, pdf) -> None:
        chunks = fixed_chunks(pdf)
        for c in chunks:
            assert f"_p{c.page}_c" in c.chunk_id

    def test_doc_id_matches_pdf(self, pdf) -> None:
        for c in fixed_chunks(pdf):
            assert c.doc_id == pdf.doc_id


class TestTokenChunks:
    def test_returns_chunks(self, pdf) -> None:
        tiktoken = pytest.importorskip("tiktoken")  # noqa: F841
        from mrta.ingestion.chunker import token_chunks

        chunks = token_chunks(pdf)
        assert len(chunks) > 0

    def test_n_tokens_within_size(self, pdf) -> None:
        pytest.importorskip("tiktoken")
        from mrta.ingestion.chunker import token_chunks

        chunks = token_chunks(pdf, size=700)
        for c in chunks:
            assert c.n_tokens is not None
            assert c.n_tokens <= 700

    def test_chunk_ids_unique(self, pdf) -> None:
        pytest.importorskip("tiktoken")
        from mrta.ingestion.chunker import token_chunks

        ids = [c.chunk_id for c in token_chunks(pdf)]
        assert len(ids) == len(set(ids))


class TestRecursiveChunks:
    def test_returns_chunks(self, pdf) -> None:
        pytest.importorskip("langchain_text_splitters")
        from mrta.ingestion.chunker import recursive_chunks

        chunks = recursive_chunks(pdf)
        assert len(chunks) > 0

    def test_chunk_ids_unique(self, pdf) -> None:
        pytest.importorskip("langchain_text_splitters")
        from mrta.ingestion.chunker import recursive_chunks

        ids = [c.chunk_id for c in recursive_chunks(pdf)]
        assert len(ids) == len(set(ids))


class TestChunkPdf:
    def test_default_strategy_is_recursive(self, pdf) -> None:
        pytest.importorskip("langchain_text_splitters")
        chunks = chunk_pdf(pdf)
        assert len(chunks) > 0

    def test_fixed_strategy(self, pdf) -> None:
        chunks = chunk_pdf(pdf, strategy="fixed")
        assert len(chunks) > 0

    def test_unknown_strategy_raises(self, pdf) -> None:
        from mrta.core.exceptions import IngestionError

        with pytest.raises(IngestionError, match="Unknown strategy"):
            chunk_pdf(pdf, strategy="nonexistent")
