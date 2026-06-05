"""Unit tests for mrta.retrieval.embedder and mrta.retrieval.vector_store."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from mrta.core.schemas import Chunk
from mrta.ingestion.chunker import fixed_chunks
from mrta.ingestion.pdf_loader import load_pdf
from mrta.retrieval.embedder import Embedder
from mrta.retrieval.vector_store import VectorStore

FIXTURE_PDF = Path(__file__).parents[1] / "fixtures" / "sample.pdf"
QUERY = "What is attention?"


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    # test.yaml sets embedding_model = sentence-transformers/all-MiniLM-L6-v2
    return Embedder()


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    return fixed_chunks(load_pdf(FIXTURE_PDF))


@pytest.fixture(scope="module")
def store(embedder: Embedder, chunks: list[Chunk]) -> VectorStore:
    vs = VectorStore(embedder)
    vs.add(chunks)
    return vs


class TestEmbedder:
    def test_embed_returns_correct_shape(self, embedder: Embedder) -> None:
        result = embedder.embed(["hello world"])
        assert result.shape == (1, embedder.dim)

    def test_embed_float32(self, embedder: Embedder) -> None:
        result = embedder.embed(["hello"])
        assert result.dtype == np.float32

    def test_embed_normalised(self, embedder: Embedder) -> None:
        result = embedder.embed(["hello", "world"])
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_dim_positive(self, embedder: Embedder) -> None:
        assert embedder.dim > 0

    def test_model_name_accessible(self, embedder: Embedder) -> None:
        assert embedder.model_name != ""


class TestVectorStore:
    def test_search_returns_k_results(self, store: VectorStore) -> None:
        results = store.search(QUERY, k=1)
        assert len(results) == 1

    def test_search_returns_chunk_instances(self, store: VectorStore) -> None:
        for c in store.search(QUERY, k=3):
            assert isinstance(c, Chunk)

    def test_search_chunks_have_text(self, store: VectorStore) -> None:
        for c in store.search(QUERY, k=3):
            assert c.text.strip()

    def test_search_k_capped_at_index_size(
        self, embedder: Embedder, chunks: list[Chunk]
    ) -> None:
        small_store = VectorStore(embedder)
        small_store.add(chunks[:2])
        results = small_store.search(QUERY, k=10)
        assert len(results) == 2

    def test_save_load_roundtrip(
        self, store: VectorStore, embedder: Embedder, tmp_path: Path
    ) -> None:
        store.save(tmp_path / "vs")
        reloaded = VectorStore.load(tmp_path / "vs", embedder)
        original_top = store.search(QUERY, k=1)
        reloaded_top = reloaded.search(QUERY, k=1)
        assert len(reloaded_top) == 1
        assert reloaded_top[0].chunk_id == original_top[0].chunk_id

    def test_save_creates_expected_files(
        self, store: VectorStore, tmp_path: Path
    ) -> None:
        store.save(tmp_path / "vs2")
        assert (tmp_path / "vs2" / "index.faiss").exists()
        assert (tmp_path / "vs2" / "metadata.jsonl").exists()
        assert (tmp_path / "vs2" / "config.json").exists()

    def test_add_empty_is_noop(self, embedder: Embedder) -> None:
        vs = VectorStore(embedder)
        vs.add([])  # must not raise
        assert vs.search(QUERY, k=3) == []
