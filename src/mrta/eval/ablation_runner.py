"""mrta.eval.ablation_runner — executes the frozen ablation matrix.

Separated from ``ablation.py`` so the metric/​config layer stays importable (and
unit-testable) without pulling in stores, models or the production pipeline.

The runner owns store access and stage sequencing; ``ablation.py`` owns the
configuration matrix, metric computation and failure attribution.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from mrta.eval.ablation import (
    DEFAULT_CANDIDATE_DEPTH,
    DEFAULT_FINAL_TOP_K,
    DEFAULT_RRF_K,
    STATUS_ERROR,
    STREAM_CAPTION,
    STREAM_CLIP,
    STREAM_TEXT,
    AblationConfig,
    QueryResult,
    classify_failure,
    compute_retrieval_metrics,
    evidence_as_dicts,
    figure_provenance,
    parse_targets,
    target_rank,
)
from mrta.eval.types import CanonicalEvidence, RetrievedCandidate


class Generator(Protocol):
    """Minimal generation surface the runner needs.

    A Protocol rather than a concrete client so tests can inject a deterministic
    fake and the deterministic suite never requires a local model.
    """

    def generate(self, prompt: str, images: list) -> str: ...


@dataclass
class RunnerStores:
    """The retrieval indices an ablation run reads from.

    All three are optional: a run limited to ``text_only`` needs no CLIP index,
    and ``oracle_evidence_generation`` needs none at all.
    """

    text: Any | None = None
    caption: Any | None = None
    clip: Any | None = None
    adapter: Any | None = None  # EvalAdapter, for benchmark->canonical mapping


def build_figure_text_lookup(caption_store: Any, adapter: Any) -> dict[str, dict]:
    """canonical_id -> figure text, built once over the whole caption index.

    Reproduces the PR5 candidate-representation policy exactly, and the policy
    matters for two reasons:

    1. A ``VisualRecord`` from the CLIP index carries no text at all, so a
       figure found only by CLIP would otherwise reach the cross-encoder as an
       empty string. PR5 gave those figures the caption index's text for the
       same canonical figure; not doing so changes the reranked ordering.
    2. A canonical figure owning several caption records (multi-crop) must
       resolve deterministically — prefer a record with a VLM caption, then the
       lowest evidence_id — rather than taking whichever crop the search
       happened to return, which is query-dependent and therefore unstable.
    """
    from mrta.retrieval.fusion import canonical_identity

    records = getattr(caption_store, "_records", [])
    grouped: dict[str, list] = {}
    for record in records:
        candidate = adapter.from_caption_record(record, 0.0, 1)
        if candidate.evidence.figure_id is None:
            continue
        grouped.setdefault("|".join(canonical_identity(candidate)), []).append(record)

    lookup: dict[str, dict] = {}
    for canonical_id, group in grouped.items():
        chosen = sorted(
            group,
            key=lambda r: (
                0 if (r.caption or r.detailed_description) else 1,
                r.evidence_id,
            ),
        )[0]
        lookup[canonical_id] = {
            "figure_caption": chosen.caption,
            "figure_description": chosen.detailed_description,
            "figure_nearby_text": chosen.nearby_text,
        }
    return lookup


def build_oracle_text_lookup(
    text_store: Any,
    adapter: Any,
    figure_text: dict[str, dict],
) -> dict[tuple, dict]:
    """CanonicalEvidence.key() -> evidence text, for the oracle configuration.

    Text targets resolve to every indexed chunk on that document page, joined in
    index order; figure targets reuse the canonical figure-text lookup. Built
    from the same indices retrieval reads, so the oracle differs from a retrieval
    configuration only in *which* evidence is selected, never in how it is
    rendered — which is what makes the comparison interpretable.
    """
    lookup: dict[tuple, dict] = {}

    by_page: dict[tuple, list[str]] = {}
    for chunk in getattr(text_store, "_chunks", []):
        candidate = adapter.chunk_to_candidate(chunk, 0.0, 1)
        by_page.setdefault(candidate.evidence.key(), []).append(chunk.text)
    for key, texts in by_page.items():
        lookup[key] = {"chunk_text": "\n".join(texts)}

    for canonical_id, payload in figure_text.items():
        _, document_id, tail = canonical_id.split("|", 2)
        page_part, figure_id = tail.split(":", 1)
        key = (document_id, int(page_part.lstrip("p")), figure_id)
        lookup[key] = payload

    return lookup


class AblationRunner:
    """Runs configurations over a frozen query set.

    One runner instance serves every configuration so the per-stream candidate
    pools are retrieved once per query and reused. Retrieval is the expensive
    deterministic stage, and re-running it per configuration would multiply cost
    by the number of configurations without changing any result.
    """

    def __init__(
        self,
        stores: RunnerStores,
        *,
        reranker: Any | None = None,
        generator: Generator | None = None,
        candidate_depth: int = DEFAULT_CANDIDATE_DEPTH,
        rrf_k: int = DEFAULT_RRF_K,
        final_top_k: int = DEFAULT_FINAL_TOP_K,
    ) -> None:
        self._stores = stores
        self._figure_text = (
            build_figure_text_lookup(stores.caption, stores.adapter)
            if (stores.caption is not None and stores.adapter is not None)
            else {}
        )
        # Ground-truth evidence must reach the generator as actual text, not as
        # a bare identifier: an oracle fed empty context measures nothing about
        # the generation stage, which is the only thing it exists to measure.
        self._oracle_text = (
            build_oracle_text_lookup(stores.text, stores.adapter, self._figure_text)
            if (stores.text is not None and stores.adapter is not None)
            else {}
        )
        self._reranker = reranker
        self._generator = generator
        self._candidate_depth = candidate_depth
        self._rrf_k = rrf_k
        self._final_top_k = final_top_k

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_streams(self, query: str) -> tuple[dict[str, list[RetrievedCandidate]], dict]:
        """Retrieve every available stream's candidate pool once.

        Canonical mapping goes through ``EvalAdapter``, not the production
        adapter in ``canonical_pipeline``. The two disagree on ``document_id``
        by design: production uses the content-hashed ``doc_id``
        ("attention_is_all_you_need_a639448e61"), while benchmark targets use
        the manifest id ("attention_is_all_you_need"). Scoring production ids
        against manifest targets matches nothing, so every metric would read
        0.0 while retrieval was in fact working. EvalAdapter resolves the
        manifest id from the chunk's source filename, which is exactly what
        PR1-PR5 scored against.

        Returns (stream name -> candidates, per-stream latency in ms).
        """
        from mrta.core.schemas import VisualRecord

        adapter = self._stores.adapter
        if adapter is None:
            raise RuntimeError(
                "RunnerStores.adapter (EvalAdapter) is required: benchmark targets use "
                "manifest document ids, which only the adapter can resolve"
            )

        pools: dict[str, list[RetrievedCandidate]] = {}
        latency: dict[str, float] = {}
        self._payload_source: dict[str, Any] = {}

        if self._stores.text is not None:
            start = time.perf_counter()
            hits = self._stores.text.search_with_scores(query, k=self._candidate_depth)
            latency[STREAM_TEXT] = (time.perf_counter() - start) * 1000
            pools[STREAM_TEXT] = [
                adapter.chunk_to_candidate(chunk, score, i + 1)
                for i, (chunk, score) in enumerate(hits)
            ]
            for chunk, _ in hits:
                self._payload_source[chunk.chunk_id] = {"chunk_text": chunk.text}

        if self._stores.caption is not None:
            start = time.perf_counter()
            hits = self._stores.caption.search_with_scores(query, k=self._candidate_depth)
            latency[STREAM_CAPTION] = (time.perf_counter() - start) * 1000
            pools[STREAM_CAPTION] = [
                adapter.from_caption_record(rec, score, i + 1)
                for i, (rec, score) in enumerate(hits)
            ]
            # Figure text comes from the canonical lookup, not the returned record:
            # multi-crop figures must resolve the same way regardless of query.
            for candidate in pools[STREAM_CAPTION]:
                self._payload_source[candidate.candidate_id] = self._figure_text_for(candidate)

        if self._stores.clip is not None:
            start = time.perf_counter()
            hits = list(self._stores.clip.search(query, top_k=self._candidate_depth))
            latency[STREAM_CLIP] = (time.perf_counter() - start) * 1000
            pools[STREAM_CLIP] = [
                (
                    adapter.from_visual_record(rec, score, i + 1)
                    if isinstance(rec, VisualRecord)
                    else adapter.from_caption_record(rec, score, i + 1)
                )
                for i, (rec, score) in enumerate(hits)
            ]
            for candidate in pools[STREAM_CLIP]:
                self._payload_source.setdefault(
                    candidate.candidate_id, self._figure_text_for(candidate)
                )

        return pools, latency

    def _figure_text_for(self, candidate: RetrievedCandidate) -> dict:
        """Canonical figure text for a candidate, shared by both visual streams."""
        from mrta.retrieval.fusion import canonical_identity

        return self._figure_text.get("|".join(canonical_identity(candidate)), {})

    def run_configuration(
        self,
        config: AblationConfig,
        query: dict,
        pools: dict[str, list[RetrievedCandidate]],
        stream_latency: dict,
        *,
        generate: bool = False,
    ) -> QueryResult:
        """Score one configuration for one query."""
        targets = parse_targets(query)
        result = QueryResult(
            query_id=query["query_id"],
            config_id=config.config_id,
            document_id=query.get("document_id"),
            intent=query.get("intent"),
            difficulty=query.get("difficulty"),
            retrieval_challenge=query.get("retrieval_challenge"),
            expected_evidence=evidence_as_dicts(targets),
        )

        try:
            if config.oracle_evidence:
                self._run_oracle(config, query, targets, result, generate=generate)
            else:
                self._run_retrieval(
                    config, query, targets, pools, stream_latency, result, generate=generate
                )
        except Exception as exc:  # noqa: BLE001 — one query must not end the run
            result.status = STATUS_ERROR
            result.error_type = type(exc).__name__
            result.error_message = str(exc)[:500]

        return result

    def _run_retrieval(
        self,
        config: AblationConfig,
        query: dict,
        targets: Sequence[CanonicalEvidence],
        pools: dict[str, list[RetrievedCandidate]],
        stream_latency: dict,
        result: QueryResult,
        *,
        generate: bool,
    ) -> None:
        missing = [s for s in config.streams if s not in pools]
        if missing:
            raise RuntimeError(f"required stream(s) unavailable: {', '.join(missing)}")

        selected = {s: pools[s] for s in config.streams}
        result.latency_ms["retrieval"] = round(
            sum(stream_latency.get(s, 0.0) for s in config.streams), 3
        )

        has_figures = any(t.figure_id is not None for t in targets)
        fused_candidates: list[RetrievedCandidate]
        payload_by_id: dict[str, dict] = {}

        if config.fuse:
            fused_candidates, payload_by_id = self._fuse(selected, result)
        else:
            # Single stream: its own ranking is the result, exactly as PR1-PR3 scored it.
            fused_candidates = list(next(iter(selected.values())))
            payload_by_id = {
                c.candidate_id: self._payload_source.get(c.candidate_id, {})
                for c in fused_candidates
            }
            result.latency_ms["fusion"] = 0.0

        pool_slice = fused_candidates[: self._candidate_depth]
        result.pool_target_rank = target_rank(pool_slice, targets, figures_only=has_figures)

        if config.rerank:
            fused_candidates = self._rerank(query["query"], pool_slice, result)
        else:
            result.latency_ms["rerank"] = 0.0

        final = fused_candidates[: self._final_top_k]
        result.final_target_rank = target_rank(fused_candidates, targets, figures_only=has_figures)
        result.final_evidence = evidence_as_dicts([c.evidence for c in final])
        result.retrieval_metrics = compute_retrieval_metrics(
            fused_candidates, targets, self._final_top_k
        )
        result.top_retrieved_figure_provenance = self._provenance_for(final, payload_by_id)
        result.target_figure_provenance = self._target_provenance(targets)

        if generate and self._generator is not None:
            self._generate_and_score(query, targets, final, payload_by_id, result)

        self._finalize(result, targets, generation_ran=generate and self._generator is not None)

    def _fuse(
        self,
        selected: dict[str, list[RetrievedCandidate]],
        result: QueryResult,
    ) -> tuple[list[RetrievedCandidate], dict[str, dict]]:
        """Canonical RRF (PR4), unmodified."""
        from mrta.retrieval.fusion import canonical_identity, reciprocal_rank_fusion_canonical

        # Carry each candidate's text through fusion, keyed by canonical id, so
        # generation can render evidence without re-querying the stores.
        payloads: dict[str, dict[str, dict]] = {}
        for stream_name, candidates in selected.items():
            payloads[stream_name] = {
                "|".join(canonical_identity(c)): self._payload_source.get(c.candidate_id, {})
                for c in candidates
            }

        start = time.perf_counter()
        fused = reciprocal_rank_fusion_canonical(
            selected, k=self._rrf_k, top_k=None, payloads=payloads
        )
        result.latency_ms["fusion"] = round((time.perf_counter() - start) * 1000, 3)

        candidates = [
            RetrievedCandidate(
                candidate_id=fc.canonical_id,
                evidence=CanonicalEvidence(
                    document_id=fc.document_id,
                    page_number=fc.page,
                    figure_id=fc.figure_id,
                ),
                score=fc.score,
                rank=i + 1,
            )
            for i, fc in enumerate(fused)
        ]
        payloads = {fc.canonical_id: dict(fc.payload) for fc in fused}
        self._last_fused = fused
        return candidates, payloads

    def _rerank(
        self,
        query_text: str,
        pool: Sequence[RetrievedCandidate],
        result: QueryResult,
    ) -> list[RetrievedCandidate]:
        """Cross-encoder reranking (PR5), unmodified."""
        if self._reranker is None:
            raise RuntimeError("configuration requires a reranker but none was supplied")

        fused = getattr(self, "_last_fused", None)
        if fused is None:
            raise RuntimeError("reranking requires a fused candidate pool")

        start = time.perf_counter()
        reranked = self._reranker.rerank(query_text, list(fused)[: len(pool)], top_k=len(pool))
        result.latency_ms["rerank"] = round((time.perf_counter() - start) * 1000, 3)

        return [
            RetrievedCandidate(
                candidate_id=rc.candidate.canonical_id,
                evidence=CanonicalEvidence(
                    document_id=rc.candidate.document_id,
                    page_number=rc.candidate.page,
                    figure_id=rc.candidate.figure_id,
                ),
                score=rc.reranker_score,
                rank=rc.reranker_rank,
            )
            for rc in reranked
        ]

    def _run_oracle(
        self,
        config: AblationConfig,
        query: dict,
        targets: Sequence[CanonicalEvidence],
        result: QueryResult,
        *,
        generate: bool,
    ) -> None:
        """Hand the generator the ground-truth evidence directly.

        Retrieval metrics are perfect by construction, which is why this
        configuration must never be mixed into a retrieval comparison. Its only
        purpose is bounding what the generation stage can do when retrieval is
        not the limiting factor.
        """
        oracle = [
            RetrievedCandidate(
                candidate_id=f"oracle_{i}",
                evidence=target,
                score=1.0,
                rank=i + 1,
            )
            for i, target in enumerate(targets)
        ]
        result.final_evidence = evidence_as_dicts(targets)
        result.pool_target_rank = 1 if targets else None
        result.final_target_rank = 1 if targets else None
        result.retrieval_metrics = compute_retrieval_metrics(oracle, targets, self._final_top_k)
        result.target_figure_provenance = self._target_provenance(targets)
        result.latency_ms.update({"retrieval": 0.0, "fusion": 0.0, "rerank": 0.0})

        oracle_payloads = {
            candidate.candidate_id: self._oracle_text.get(candidate.evidence.key(), {})
            for candidate in oracle
        }

        if generate and self._generator is not None:
            self._generate_and_score(
                query, targets, oracle, oracle_payloads, result, oracle_mode=True
            )

        self._finalize(result, targets, generation_ran=generate and self._generator is not None)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_and_score(
        self,
        query: dict,
        targets: Sequence[CanonicalEvidence],
        final: Sequence[RetrievedCandidate],
        payload_by_id: dict[str, dict],
        result: QueryResult,
        *,
        oracle_mode: bool = False,
    ) -> None:
        from mrta.eval.generation_metrics import (
            citation_precision_recall,
            citation_validity,
            evidence_coverage,
            lexical_support_score,
            size_metrics,
            unsupported_claim_fraction,
            unsupported_numeric_tokens,
        )

        labelled = self._build_labelled_evidence(final, payload_by_id, oracle_mode=oracle_mode)
        context = self._render_context(labelled)
        prompt = self._render_prompt(query["query"], labelled)

        start = time.perf_counter()
        answer = self._generator.generate(prompt, [])  # type: ignore[union-attr]
        result.latency_ms["generation"] = round((time.perf_counter() - start) * 1000, 3)
        result.generated_answer = answer

        referenced, unknown = self._resolve_labels(answer, labelled)
        cited_evidence = [labelled[label]["evidence"] for label in referenced]

        result.resolved_citations = [
            {
                "label": label,
                **evidence_as_dicts([labelled[label]["evidence"]])[0],
            }
            for label in referenced
        ]

        scores = citation_precision_recall(cited_evidence, targets)
        validity = citation_validity(referenced, unknown)
        support = lexical_support_score(answer, context)
        claims = unsupported_claim_fraction(answer, context)
        sizes = size_metrics(context, answer)

        result.generation_metrics = {
            **scores,
            **validity.as_dict(),
            "lexical_support_score": support,
            **claims,
            **sizes,
            "unsupported_numeric_tokens": unsupported_numeric_tokens(answer, context),
            "text_evidence_count": sum(
                1 for v in labelled.values() if v["evidence"].figure_id is None
            ),
            "figure_evidence_count": sum(
                1 for v in labelled.values() if v["evidence"].figure_id is not None
            ),
            "total_evidence_count": len(labelled),
        }
        result.coverage = evidence_coverage(cited_evidence, targets)

    def _build_labelled_evidence(
        self,
        final: Sequence[RetrievedCandidate],
        payload_by_id: dict[str, dict],
        *,
        oracle_mode: bool,
    ) -> dict[str, dict]:
        """Assign [T#]/[F#] labels, mirroring the production labelling scheme."""
        labelled: dict[str, dict] = {}
        n_text = n_figure = 0
        for candidate in final:
            payload = payload_by_id.get(candidate.candidate_id, {})
            if candidate.evidence.figure_id is None:
                n_text += 1
                label = f"[T{n_text}]"
                text = payload.get("chunk_text") or ""
            else:
                n_figure += 1
                label = f"[F{n_figure}]"
                text = (
                    payload.get("figure_caption")
                    or payload.get("figure_description")
                    or payload.get("figure_nearby_text")
                    or ""
                )
            labelled[label] = {
                "evidence": candidate.evidence,
                "text": text,
                "oracle": oracle_mode,
            }
        return labelled

    def _render_context(self, labelled: dict[str, dict]) -> str:
        return "\n\n".join(
            f"{label}\nDocument: {v['evidence'].document_id} | Page: {v['evidence'].page_number}"
            f"\n{v['text']}"
            for label, v in labelled.items()
        )

    def _render_prompt(self, question: str, labelled: dict[str, dict]) -> str:
        from mrta.prompts import load_prompt

        text_evidence = [
            _PromptView(label, v) for label, v in labelled.items() if label.startswith("[T")
        ]
        figure_evidence = [
            _PromptView(label, v) for label, v in labelled.items() if label.startswith("[F")
        ]
        return load_prompt(
            "canonical_multimodal_rag",
            question=question,
            text_evidence=text_evidence,
            figure_evidence=figure_evidence,
        )

    @staticmethod
    def _resolve_labels(answer: str, labelled: dict[str, dict]) -> tuple[list[str], list[str]]:
        """Split the answer's citation labels into resolvable and unknown."""
        import re

        found = {f"[{kind}{num}]" for kind, num in re.findall(r"\[([TF])(\d+)\]", answer or "")}
        known = set(labelled)
        return sorted(found & known), sorted(found - known)

    # ------------------------------------------------------------------

    def _target_provenance(self, targets: Sequence[CanonicalEvidence]) -> str | None:
        """Provenance of the figure the benchmark expects, not the one retrieved.

        This is PR5's grouping: it asks what textual representation the *target*
        figure has, independent of whether the system found it. Grouping by the
        retrieved figure instead measures a different population and cannot be
        compared with PR5's numbers.
        """
        for target in targets:
            if target.figure_id is None:
                continue
            canonical_id = f"figure|{target.document_id}|p{target.page_number}:{target.figure_id}"
            payload = self._figure_text.get(canonical_id)
            if payload is not None:
                return figure_provenance(payload)
        return None

    def _provenance_for(
        self,
        final: Sequence[RetrievedCandidate],
        payload_by_id: dict[str, dict],
    ) -> str | None:
        """Provenance of the first figure in the final slice, if any."""
        for candidate in final:
            if candidate.evidence.figure_id is not None:
                return figure_provenance(payload_by_id.get(candidate.candidate_id))
        return None

    def _finalize(
        self,
        result: QueryResult,
        targets: Sequence[CanonicalEvidence],
        *,
        generation_ran: bool,
    ) -> None:
        gm = result.generation_metrics
        result.failure_category = classify_failure(
            targets=targets,
            pool_rank=result.pool_target_rank,
            final_rank=(
                result.final_target_rank
                if (result.final_target_rank or 0) <= self._final_top_k
                else None
            ),
            generation_ran=generation_ran,
            citation_scores=gm or None,
            validity_rate=gm.get("citation_validity_rate") if gm else None,
            support_score=gm.get("lexical_support_score") if gm else None,
        )
        result.latency_ms["total"] = round(
            sum(v for k, v in result.latency_ms.items() if k != "total"), 3
        )


@dataclass
class _PromptView:
    """Adapts labelled evidence to the fields the production template reads."""

    label: str
    _data: dict

    def __init__(self, label: str, data: dict) -> None:
        self.label = label
        self._data = data

    @property
    def source(self) -> str:
        return self._data["evidence"].document_id

    @property
    def page(self) -> int:
        return self._data["evidence"].page_number

    @property
    def figure_id(self) -> str | None:
        return self._data["evidence"].figure_id

    @property
    def text(self) -> str:
        return self._data["text"]

    @property
    def caption(self) -> str:
        return self._data["text"]


__all__ = ["AblationRunner", "Generator", "RunnerStores"]
