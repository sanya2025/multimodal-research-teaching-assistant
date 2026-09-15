"""Tests for mrta.eval.query_embedding_cache.

The committed cache is what lets CI reproduce the frozen retrieval baselines
without an Ollama server, so the tests that matter most are the integrity ones:
it must cover every benchmark query, and it must refuse to be used against a
benchmark it was not built from.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mrta.eval.query_embedding_cache import (
    CACHE_FORMAT_VERSION,
    CachedQueryEmbedder,
    QueryEmbeddingCache,
    QueryEmbeddingMiss,
    file_sha256,
    text_key,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
QUERIES_V2 = REPO_ROOT / "data" / "eval" / "queries_v2.json"
CACHE_V2 = REPO_ROOT / "data" / "eval" / "query_embeddings_v2.npz"


class _FakeEmbedder:
    """Deterministic stand-in — no server, no weights."""

    def __init__(self, dim: int = 4, name: str = "fake-embed") -> None:
        self._dim, self._name = dim, name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._name

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array(
            [np.full(self._dim, float(len(t)), dtype="float32") for t in texts], dtype="float32"
        )


@pytest.fixture
def queries_file(tmp_path: Path) -> Path:
    path = tmp_path / "queries.json"
    path.write_text(json.dumps({"queries": [{"query": "alpha"}, {"query": "beta"}]}))
    return path


@pytest.fixture
def cache(queries_file: Path) -> QueryEmbeddingCache:
    return QueryEmbeddingCache.build(
        queries=["alpha", "beta"],
        embedder=_FakeEmbedder(),
        queries_path=queries_file,
    )


class TestKeys:
    def test_text_key_is_stable(self):
        assert text_key("hello") == text_key("hello")

    def test_text_key_distinguishes_texts(self):
        assert text_key("hello") != text_key("hello ")

    def test_file_sha256_changes_with_content(self, tmp_path: Path):
        p = tmp_path / "f.json"
        p.write_text("a")
        first = file_sha256(p)
        p.write_text("b")
        assert file_sha256(p) != first


class TestCache:
    def test_build_records_model_and_dim(self, cache):
        assert cache.model_name == "fake-embed"
        assert cache.dim == 4
        assert len(cache) == 2

    def test_get_returns_the_frozen_vector(self, cache):
        np.testing.assert_array_equal(cache.get("alpha"), np.full(4, 5.0, dtype="float32"))

    def test_contains(self, cache):
        assert "alpha" in cache
        assert "gamma" not in cache

    def test_miss_raises(self, cache):
        with pytest.raises(QueryEmbeddingMiss, match="not in frozen cache"):
            cache.get("gamma")

    def test_whitespace_difference_is_a_miss(self, cache):
        """Keys are exact. A near-match must not silently resolve."""
        with pytest.raises(QueryEmbeddingMiss):
            cache.get("alpha ")

    def test_duplicate_queries_are_collapsed(self, queries_file):
        built = QueryEmbeddingCache.build(
            queries=["alpha", "alpha", "beta"],
            embedder=_FakeEmbedder(),
            queries_path=queries_file,
        )
        assert len(built) == 2

    def test_rejects_key_vector_length_mismatch(self):
        with pytest.raises(ValueError, match="keys but"):
            QueryEmbeddingCache(
                keys=["a"],
                vectors=np.zeros((2, 4), dtype="float32"),
                model_name="m",
                queries_sha256="x",
            )

    def test_rejects_non_2d_vectors(self):
        with pytest.raises(ValueError, match="2-D"):
            QueryEmbeddingCache(
                keys=["a"],
                vectors=np.zeros(4, dtype="float32"),
                model_name="m",
                queries_sha256="x",
            )

    def test_rejects_duplicate_keys(self):
        with pytest.raises(ValueError, match="duplicate"):
            QueryEmbeddingCache(
                keys=["a", "a"],
                vectors=np.zeros((2, 4), dtype="float32"),
                model_name="m",
                queries_sha256="x",
            )


class TestPersistence:
    def test_round_trip(self, cache, tmp_path):
        path = tmp_path / "c.npz"
        cache.save(path)
        loaded = QueryEmbeddingCache.load(path)
        assert len(loaded) == len(cache)
        assert loaded.model_name == cache.model_name
        assert loaded.queries_sha256 == cache.queries_sha256
        np.testing.assert_array_equal(loaded.get("beta"), cache.get("beta"))

    def test_save_is_byte_stable(self, cache, tmp_path):
        """Key-sorted output, so re-saving the same cache produces the same file."""
        a, b = tmp_path / "a.npz", tmp_path / "b.npz"
        cache.save(a)
        QueryEmbeddingCache.load(a).save(b)
        np.testing.assert_array_equal(
            QueryEmbeddingCache.load(a).get("alpha"), QueryEmbeddingCache.load(b).get("alpha")
        )

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            QueryEmbeddingCache.load(tmp_path / "absent.npz")

    def test_format_version_mismatch_raises(self, cache, tmp_path):
        path = tmp_path / "c.npz"
        cache.save(path)
        with np.load(path, allow_pickle=True) as data:
            fields = {k: data[k] for k in data.files}
        fields["format_version"] = np.asarray(CACHE_FORMAT_VERSION + 1)
        np.savez_compressed(path, **fields)
        with pytest.raises(ValueError, match="format"):
            QueryEmbeddingCache.load(path)


class TestCachedQueryEmbedder:
    def test_presents_the_embedder_interface(self, cache):
        embedder = CachedQueryEmbedder(cache)
        assert embedder.dim == 4
        assert embedder.model_name == "fake-embed"

    def test_embed_returns_stacked_float32(self, cache):
        out = CachedQueryEmbedder(cache).embed(["alpha", "beta"])
        assert out.shape == (2, 4)
        assert out.dtype == np.float32

    def test_embed_preserves_input_order(self, cache):
        out = CachedQueryEmbedder(cache).embed(["beta", "alpha"])
        np.testing.assert_array_equal(out[0], cache.get("beta"))
        np.testing.assert_array_equal(out[1], cache.get("alpha"))

    def test_embed_empty_list(self, cache):
        assert CachedQueryEmbedder(cache).embed([]).shape == (0, 4)

    def test_embed_raises_on_unknown_query(self, cache):
        """No live fallback: a miss must fail loudly, not silently re-embed."""
        with pytest.raises(QueryEmbeddingMiss):
            CachedQueryEmbedder(cache).embed(["alpha", "unknown"])

    def test_verify_queries_accepts_the_source_file(self, cache, queries_file):
        CachedQueryEmbedder(cache).verify_queries(queries_file)

    def test_verify_queries_rejects_a_changed_file(self, cache, queries_file):
        queries_file.write_text(json.dumps({"queries": [{"query": "alpha"}]}))
        with pytest.raises(ValueError, match="has changed since"):
            CachedQueryEmbedder(cache).verify_queries(queries_file)


@pytest.fixture(scope="module")
def committed() -> QueryEmbeddingCache:
    return QueryEmbeddingCache.load(CACHE_V2)


@pytest.mark.eval
class TestCommittedV2Cache:
    """Integrity of the checked-in artifact Tier 2 depends on."""

    def test_exists(self):
        assert CACHE_V2.exists(), "run scripts/build_query_embeddings.py --benchmark v2"

    def test_matches_the_index_model_and_dim(self, committed):
        index_config = json.loads(
            (REPO_ROOT / "data/eval/indices/v2/caption_index/config.json").read_text()
        )
        assert committed.model_name == index_config["model"]
        assert committed.dim == index_config["dim"]

    def test_covers_every_benchmark_query(self, committed):
        queries = [q["query"] for q in json.loads(QUERIES_V2.read_text())["queries"]]
        missing = [q for q in queries if q not in committed]
        assert not missing, f"{len(missing)} benchmark queries absent from the frozen cache"

    def test_query_count_matches_the_benchmark(self, committed):
        assert len(committed) == json.loads(QUERIES_V2.read_text())["query_count"]

    def test_bound_to_the_committed_queries_file(self, committed):
        CachedQueryEmbedder(committed).verify_queries(QUERIES_V2)

    def test_vectors_are_l2_normalised(self, committed):
        """The indices use IndexFlatIP, where inner product is cosine only if
        both sides are unit-norm. An unnormalised query vector would rank wrongly
        without ever erroring."""
        sample = committed.get(json.loads(QUERIES_V2.read_text())["queries"][0]["query"])
        assert abs(float(np.linalg.norm(sample)) - 1.0) < 1e-5


def _load_run_ablation():
    """Import scripts/run_ablation.py as a module (it is a script, not a package)."""
    import importlib.util

    path = REPO_ROOT / "scripts" / "run_ablation.py"
    spec = importlib.util.spec_from_file_location("_run_ablation_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.eval
class TestOfflineWiringIgnoresSettings:
    """The offline evaluation path must not be steerable by configuration.

    `settings.embedding_model` resolves to all-MiniLM-L6-v2 under MRTA_ENV=test
    (configs/test.yaml) and under the repo default — and Tier 2 runs with
    MRTA_ENV=test. If the offline branch ever consults settings again, the
    768-d v2 indices would be queried with 384-d vectors. These tests pin the
    invariant that the model comes from the frozen cache, nothing else.
    """

    def test_build_stores_reports_the_cache_model_not_settings(self, monkeypatch):
        pytest.importorskip("faiss")
        from mrta.core.config import settings

        run_ablation = _load_run_ablation()
        monkeypatch.setattr(settings, "embedding_model", "sentence-transformers/all-MiniLM-L6-v2")

        _, models = run_ablation.build_stores("v2", {"text", "caption"}, offline=True)
        assert models["embedding_model"] == "nomic-embed-text"

    def test_offline_stores_use_the_cached_embedder(self, monkeypatch):
        pytest.importorskip("faiss")
        from mrta.core.config import settings

        run_ablation = _load_run_ablation()
        monkeypatch.setattr(settings, "embedding_model", "sentence-transformers/all-MiniLM-L6-v2")

        stores, _ = run_ablation.build_stores("v2", {"text", "caption"}, offline=True)
        assert isinstance(stores.text._embedder, CachedQueryEmbedder)
        assert stores.text._embedder.dim == 768

    def test_offline_retrieval_actually_returns_results(self, monkeypatch):
        """End of the chain: a frozen query vector searches the committed index.

        A 384-d embedder here would trip FAISS's `assert d == self.d`, so this
        passing is positive evidence that the 768-d path is the one in use.
        """
        pytest.importorskip("faiss")
        from mrta.core.config import settings

        run_ablation = _load_run_ablation()
        monkeypatch.setattr(settings, "embedding_model", "sentence-transformers/all-MiniLM-L6-v2")

        stores, _ = run_ablation.build_stores("v2", {"text"}, offline=True)
        query = json.loads(QUERIES_V2.read_text())["queries"][0]["query"]
        assert stores.text.search_with_scores(query, k=5)
