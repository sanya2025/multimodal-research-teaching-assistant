"""mrta.retrieval.image_store — FAISS index over direct CLIP image embeddings.

Architecture::

    figure PNG  →  CLIP image encoder  →  512-D unit vector  →  IndexFlatIP
    text query  →  CLIP text encoder   →  512-D unit vector  →  index search

No captions, descriptions, or nearby text participate. This store measures what
CLIP sees in the image itself, which is what makes it a clean ablation against
caption-based retrieval.

The index is kept separate from VectorStore (nomic-embed-text) and
CaptionVectorStore (nomic-embed-text) because CLIP vectors occupy a different
embedding space. Raw scores across those spaces are not calibrated against each
other and must not be pooled into one ranking; combining streams is a rank-fusion
concern, not a scoring one.

Persistence follows the convention established by CaptionVectorStore: the index
stores lightweight metadata plus an ``image_path`` reference, never raw image bytes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mrta.core.exceptions import RetrievalError
from mrta.core.schemas import VisualRecord
from mrta.retrieval.clip_embedder import CLIPEmbedder

if TYPE_CHECKING:
    import faiss


class ImageStore:
    """IndexFlatIP FAISS index over CLIP image embeddings of figure PNGs.

    All vectors are unit-normalized, so inner product is cosine similarity.
    """

    def __init__(self, embedder: CLIPEmbedder) -> None:
        self._embedder = embedder
        self._index: faiss.Index | None = None
        self._records: list[VisualRecord] = []

    def _ensure_index(self) -> faiss.Index:
        if self._index is None:
            # Torch must initialize its OpenMP runtime before FAISS does, or the
            # first torch forward pass after FAISS loads segfaults on macOS.
            # See CLIPEmbedder.warmup().
            self._embedder.warmup()
            import faiss

            self._index = faiss.IndexFlatIP(self._embedder.dim)
        return self._index

    @property
    def size(self) -> int:
        """Number of records currently indexed."""
        return len(self._records)

    @property
    def records(self) -> list[VisualRecord]:
        """Indexed records in insertion order (deterministic)."""
        return list(self._records)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_images(self, records: list[VisualRecord]) -> None:
        """Embed each record's image and add it to the index.

        Records are added in the given order and that order is preserved on save,
        so a rebuilt index is byte-comparable given the same inputs.

        Raises:
            FileNotFoundError: a record's image_path does not exist. Failing loudly
                is deliberate — a silently skipped figure would understate recall
                and be very hard to notice in aggregate metrics.
        """
        if not records:
            return
        vectors = [self._embedder.embed_image(rec.image_path) for rec in records]
        matrix = np.stack(vectors).astype("float32")
        self._ensure_index().add(matrix)
        self._records.extend(records)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[tuple[VisualRecord, float]]:
        """Return up to top_k (record, cosine_score) pairs, highest score first.

        Returned records are copies carrying retrieval_score; stored records are
        never mutated. If top_k exceeds the index size, every record is returned.
        """
        if not self._records:
            return []
        query_vec = self._embedder.embed_text(query).reshape(1, -1).astype("float32")
        fetch_k = min(top_k, len(self._records))
        scores, indices = self._ensure_index().search(query_vec, fetch_k)

        results: list[tuple[VisualRecord, float]] = []
        for rank, idx in enumerate(indices[0]):
            if not (0 <= idx < len(self._records)):
                continue  # FAISS pads with -1 when fewer than fetch_k results exist
            record = self._records[idx]
            score = float(scores[0][rank])
            results.append((record.model_copy(update={"retrieval_score": score}), score))
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Write index.faiss + metadata.jsonl + config.json to path.

        config.json records the model identifier, dimension, normalization, and
        similarity convention so a reloaded index can be verified against the
        embedder it is paired with.
        """
        import faiss

        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._ensure_index(), str(p / "index.faiss"))
        (p / "metadata.jsonl").write_text(
            "\n".join(r.model_dump_json() for r in self._records),
            encoding="utf-8",
        )
        (p / "config.json").write_text(
            json.dumps(
                {
                    "model": self._embedder.model_name,
                    "embedding_dimension": self._embedder.dim,
                    "normalization": "L2",
                    "similarity": "inner_product/cosine",
                    "index_type": "IndexFlatIP",
                    "image_bytes_persisted": False,
                    "n_records": len(self._records),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str, embedder: CLIPEmbedder) -> ImageStore:
        """Reload a persisted store.

        Raises:
            RetrievalError: the FAISS index is unreadable, or the persisted model
                identifier does not match the supplied embedder — comparing vectors
                produced by different models would yield meaningless scores.
        """
        # Warm torch before FAISS loads — see CLIPEmbedder.warmup(). This is the
        # first FAISS touch in most PR3 code paths, so the ordering is set here.
        embedder.warmup()
        import faiss

        p = Path(path)
        store = cls(embedder)
        try:
            store._index = faiss.read_index(str(p / "index.faiss"))
        except Exception as e:
            raise RetrievalError(f"Cannot load FAISS index from {p}: {e}") from e

        config_path = p / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            saved_model = config.get("model")
            if saved_model and saved_model != embedder.model_name:
                raise RetrievalError(
                    f"Index at {p} was built with model {saved_model!r} but the "
                    f"supplied embedder uses {embedder.model_name!r}. "
                    "Embeddings from different models are not comparable."
                )

        lines = (p / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        store._records = [VisualRecord.model_validate_json(line) for line in lines if line]
        return store
