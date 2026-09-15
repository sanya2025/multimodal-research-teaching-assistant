"""Unit tests for PR8 generation conditions and experimental integrity.

Fully mocked: no Ollama, no network, no GPU, no model download.
"""

from __future__ import annotations

import pytest

from mrta.eval.generation_conditions import (
    CONDITIONS_BY_ID,
    FAILURE_EMPTY_ANSWER,
    FAILURE_INVALID_CITATION,
    FAILURE_IRRELEVANT_VALID_CITATION,
    FAILURE_MALFORMED_STRUCTURED_OUTPUT,
    FAILURE_MISSING_EXPECTED_CITATION,
    FAILURE_UNCITED_CLAIM,
    FROZEN_CONDITIONS,
    G0_BASELINE,
    G1_EXPLICIT_CITATIONS,
    G2_STRUCTURED_EVIDENCE,
    G3_STRUCTURED_GENERATION,
    claim_citation_coverage,
    classify_generation_failures,
    evidence_context_hash,
    extract_inline_labels,
    paired_comparison,
    parse_generation,
    parse_structured_generation,
)
from mrta.eval.types import CanonicalEvidence
from mrta.prompts import load_prompt

DOC = "attention_is_all_you_need"


class EvidenceView:
    """Minimal stand-in for the prompt's evidence view."""

    def __init__(self, label, source, page, text="", caption=None, figure_id=None):
        self.label = label
        self.source = source
        self.page = page
        self.text = text
        self.caption = caption
        self.figure_id = figure_id


def _text_view(label="[T1]", text="Attention uses softmax scaling."):
    return EvidenceView(label, "attention.pdf", 2, text=text)


def _figure_view(label="[F1]", caption="Diagram of the Transformer."):
    return EvidenceView(label, "attention.pdf", 4, caption=caption, figure_id="p4_f1")


def _render(condition_id: str, text_evidence=None, figure_evidence=None) -> str:
    return load_prompt(
        CONDITIONS_BY_ID[condition_id].template,
        question="How is attention computed?",
        text_evidence=text_evidence if text_evidence is not None else [_text_view()],
        figure_evidence=figure_evidence if figure_evidence is not None else [_figure_view()],
    )


def _labelled(*, with_figure: bool = True) -> dict:
    entries = {
        "[T1]": {
            "evidence": CanonicalEvidence(DOC, 2, None),
            "text": "Attention uses softmax scaling.",
        }
    }
    if with_figure:
        entries["[F1]"] = {
            "evidence": CanonicalEvidence(DOC, 4, "p4_f1"),
            "text": "Diagram of the Transformer.",
        }
    return entries


# ---------------------------------------------------------------------------
# Conditions and prompts (spec section 39)
# ---------------------------------------------------------------------------


class TestConditionDefinitions:
    def test_four_frozen_conditions(self) -> None:
        assert {c.condition_id for c in FROZEN_CONDITIONS} == {
            G0_BASELINE,
            G1_EXPLICIT_CITATIONS,
            G2_STRUCTURED_EVIDENCE,
            G3_STRUCTURED_GENERATION,
        }

    def test_only_g3_requests_structured_output(self) -> None:
        structured = {c.condition_id for c in FROZEN_CONDITIONS if c.structured_output}
        assert structured == {G3_STRUCTURED_GENERATION}

    def test_g0_uses_the_unchanged_production_template(self) -> None:
        """G0 must be the PR6/PR7 baseline, or it is not a baseline."""
        assert CONDITIONS_BY_ID[G0_BASELINE].template == "canonical_multimodal_rag"

    def test_prompt_hashes_are_stable_and_distinct(self) -> None:
        hashes = {c.condition_id: c.prompt_hash() for c in FROZEN_CONDITIONS}
        assert all(h != "missing" for h in hashes.values())
        assert len(set(hashes.values())) == len(hashes)
        # stable across calls
        assert hashes == {c.condition_id: c.prompt_hash() for c in FROZEN_CONDITIONS}


class TestPromptRendering:
    @pytest.mark.parametrize("condition_id", list(CONDITIONS_BY_ID))
    def test_every_condition_renders(self, condition_id: str) -> None:
        assert _render(condition_id).strip()

    @pytest.mark.parametrize("condition_id", list(CONDITIONS_BY_ID))
    def test_supplied_labels_appear(self, condition_id: str) -> None:
        rendered = _render(condition_id)
        assert "[T1]" in rendered
        assert "[F1]" in rendered

    @pytest.mark.parametrize("condition_id", list(CONDITIONS_BY_ID))
    def test_text_and_figure_evidence_are_separated(self, condition_id: str) -> None:
        rendered = _render(condition_id).upper()
        assert "TEXT EVIDENCE" in rendered
        assert "FIGURE EVIDENCE" in rendered
        assert rendered.index("TEXT EVIDENCE") < rendered.index("FIGURE EVIDENCE")

    @pytest.mark.parametrize("condition_id", list(CONDITIONS_BY_ID))
    def test_image_path_is_never_prompt_content(self, condition_id: str) -> None:
        """A filesystem path carries no visual information and must not appear."""
        view = _figure_view()
        view.image_path = "data/figures/secret_figure.png"  # type: ignore[attr-defined]
        rendered = _render(condition_id, figure_evidence=[view])
        assert "secret_figure.png" not in rendered
        assert "data/figures" not in rendered

    @pytest.mark.parametrize("condition_id", list(CONDITIONS_BY_ID))
    def test_no_benchmark_ground_truth_in_prompt(self, condition_id: str) -> None:
        rendered = _render(condition_id)
        for leaked in (
            "expected_evidence",
            "source_note",
            "retrieval_challenge",
            "difficulty",
            "target_evidence",
        ):
            assert leaked not in rendered

    def test_g1_adds_citation_requirements_over_g0(self) -> None:
        g0, g1 = _render(G0_BASELINE), _render(G1_EXPLICIT_CITATIONS)
        assert "Citation requirements" not in g0
        assert "Citation requirements" in g1

    def test_g2_uses_evidence_cards(self) -> None:
        rendered = _render(G2_STRUCTURED_EVIDENCE)
        assert "[card [T1]]" in rendered
        assert "document:" in rendered and "content:" in rendered

    def test_g3_specifies_the_json_contract(self) -> None:
        rendered = _render(G3_STRUCTURED_GENERATION)
        assert '"answer"' in rendered and '"citations"' in rendered
        assert "RESPONSE FORMAT" in rendered

    def test_no_figures_renders_without_figure_section(self) -> None:
        """The section delimiter is absent; the preamble may still explain cards."""
        rendered = _render(G2_STRUCTURED_EVIDENCE, figure_evidence=[])
        assert "=== FIGURE EVIDENCE ===" not in rendered
        assert "[card [F" not in rendered

    def test_empty_evidence_says_so(self) -> None:
        rendered = _render(G2_STRUCTURED_EVIDENCE, text_evidence=[], figure_evidence=[])
        assert "no evidence" in rendered.lower()


# ---------------------------------------------------------------------------
# Structured output parsing (spec sections 25, 39)
# ---------------------------------------------------------------------------


class TestStructuredParsing:
    def test_valid_json_parses(self) -> None:
        parsed = parse_structured_generation(
            '{"answer": "Attention uses softmax [T1].", "citations": ["T1", "F1"]}'
        )
        assert parsed.malformed is False
        assert parsed.labels == ["[T1]", "[F1]"]
        assert parsed.answer == "Attention uses softmax [T1]."

    def test_fenced_json_parses(self) -> None:
        parsed = parse_structured_generation(
            '```json\n{"answer": "A [T1].", "citations": ["T1"]}\n```'
        )
        assert parsed.malformed is False
        assert parsed.labels == ["[T1]"]

    def test_bracketed_labels_accepted(self) -> None:
        parsed = parse_structured_generation('{"answer": "A.", "citations": ["[T1]", "[F2]"]}')
        assert parsed.labels == ["[T1]", "[F2]"]

    def test_lowercase_labels_normalised(self) -> None:
        parsed = parse_structured_generation('{"answer": "A.", "citations": ["t1", " f2 "]}')
        assert parsed.labels == ["[T1]", "[F2]"]

    def test_duplicate_citations_deduplicated(self) -> None:
        parsed = parse_structured_generation('{"answer": "A.", "citations": ["T1", "T1", "[T1]"]}')
        assert parsed.labels == ["[T1]"]

    def test_non_label_citations_rejected(self) -> None:
        """The model must not be able to inject paths or page numbers as citations."""
        parsed = parse_structured_generation(
            '{"answer": "A.", "citations": ["page 4", "/etc/passwd", "T1", "figure 2"]}'
        )
        assert parsed.labels == ["[T1]"]

    def test_invalid_json_degrades_safely(self) -> None:
        parsed = parse_structured_generation("not json at all [T2]")
        assert parsed.malformed is True
        assert parsed.labels == ["[T2]"]  # falls back to inline extraction
        assert "invalid JSON" in (parsed.parse_note or "")

    def test_json_array_is_malformed(self) -> None:
        parsed = parse_structured_generation("[1, 2, 3]")
        assert parsed.malformed is True
        assert "expected object" in (parsed.parse_note or "")

    def test_missing_answer_field_is_malformed(self) -> None:
        parsed = parse_structured_generation('{"citations": ["T1"]}')
        assert parsed.malformed is True

    def test_citations_not_a_list_keeps_answer(self) -> None:
        parsed = parse_structured_generation('{"answer": "A [T1].", "citations": "T1"}')
        assert parsed.malformed is True
        assert parsed.answer == "A [T1]."
        assert parsed.labels == ["[T1]"]

    def test_empty_reply_is_malformed(self) -> None:
        parsed = parse_structured_generation("")
        assert parsed.malformed is True
        assert parsed.labels == []

    def test_empty_citations_list_falls_back_to_inline(self) -> None:
        parsed = parse_structured_generation('{"answer": "A [T1].", "citations": []}')
        assert parsed.malformed is False
        assert parsed.labels == ["[T1]"]

    def test_prose_conditions_use_inline_extraction(self) -> None:
        parsed = parse_generation(
            "Softmax [T1] and the diagram [F1].", CONDITIONS_BY_ID[G0_BASELINE]
        )
        assert parsed.labels == ["[T1]", "[F1]"]
        assert parsed.malformed is False

    def test_inline_extraction_deduplicates_preserving_order(self) -> None:
        assert extract_inline_labels("[F2] then [T1] then [F2]") == ["[F2]", "[T1]"]

    def test_inline_extraction_on_empty(self) -> None:
        assert extract_inline_labels("") == []


# ---------------------------------------------------------------------------
# Claim-level citation coverage (spec section 13)
# ---------------------------------------------------------------------------


class TestClaimCitationCoverage:
    def test_all_claims_cited(self) -> None:
        result = claim_citation_coverage("Attention uses softmax [T1]. The diagram shows it [F1].")
        assert result["claim_citation_coverage"] == 1.0
        assert result["uncited_claim_fraction"] == 0.0
        assert result["claim_count"] == 2

    def test_partial_coverage(self) -> None:
        result = claim_citation_coverage("Attention uses softmax [T1]. Something else entirely.")
        assert result["claim_citation_coverage"] == pytest.approx(0.5)

    def test_no_citations(self) -> None:
        result = claim_citation_coverage("A claim. Another claim.")
        assert result["claim_citation_coverage"] == 0.0
        assert result["uncited_claim_fraction"] == 1.0

    def test_empty_answer_reports_none(self) -> None:
        result = claim_citation_coverage("")
        assert result["claim_count"] == 0
        assert result["claim_citation_coverage"] is None

    def test_non_alphabetic_fragments_skipped(self) -> None:
        result = claim_citation_coverage("Attention uses softmax [T1]. 123. ...")
        assert result["claim_count"] == 1
        assert result["claim_citation_coverage"] == 1.0


# ---------------------------------------------------------------------------
# Evidence-context hash — the experimental-integrity primitive (section 20)
# ---------------------------------------------------------------------------


class TestEvidenceContextHash:
    def test_identical_evidence_gives_identical_hash(self) -> None:
        assert evidence_context_hash(_labelled()) == evidence_context_hash(_labelled())

    def test_hash_is_order_independent(self) -> None:
        forward = _labelled()
        reversed_order = dict(reversed(list(forward.items())))
        assert evidence_context_hash(forward) == evidence_context_hash(reversed_order)

    def test_changed_evidence_text_changes_hash(self) -> None:
        modified = _labelled()
        modified["[T1]"]["text"] = "Completely different passage text."
        assert evidence_context_hash(modified) != evidence_context_hash(_labelled())

    def test_changed_canonical_identity_changes_hash(self) -> None:
        modified = _labelled()
        modified["[T1]"]["evidence"] = CanonicalEvidence(DOC, 99, None)
        assert evidence_context_hash(modified) != evidence_context_hash(_labelled())

    def test_dropped_evidence_changes_hash(self) -> None:
        assert evidence_context_hash(_labelled(with_figure=False)) != evidence_context_hash(
            _labelled()
        )

    def test_empty_evidence_is_stable(self) -> None:
        assert evidence_context_hash({}) == evidence_context_hash({})


# ---------------------------------------------------------------------------
# Generation failure taxonomy (spec section 36)
# ---------------------------------------------------------------------------


class TestGenerationFailureTaxonomy:
    def _classify(self, **overrides):
        kwargs = {
            "answer": "An answer [T1].",
            "citation_scores": {"citation_recall": 1.0, "citation_precision": 1.0},
            "invalid_count": 0,
            "malformed": False,
            "claim_coverage": 1.0,
            "has_targets": True,
        }
        kwargs.update(overrides)
        return classify_generation_failures(**kwargs)

    def test_clean_answer_has_no_failures(self) -> None:
        assert self._classify() == []

    def test_missing_expected_citation(self) -> None:
        assert FAILURE_MISSING_EXPECTED_CITATION in self._classify(
            citation_scores={"citation_recall": 0.5, "citation_precision": 1.0}
        )

    def test_irrelevant_valid_citation(self) -> None:
        assert FAILURE_IRRELEVANT_VALID_CITATION in self._classify(
            citation_scores={"citation_recall": 1.0, "citation_precision": 0.5}
        )

    def test_invalid_citation(self) -> None:
        assert FAILURE_INVALID_CITATION in self._classify(invalid_count=1)

    def test_malformed_structured_output(self) -> None:
        assert FAILURE_MALFORMED_STRUCTURED_OUTPUT in self._classify(malformed=True)

    def test_uncited_claims(self) -> None:
        assert FAILURE_UNCITED_CLAIM in self._classify(claim_coverage=0.1)

    def test_empty_answer(self) -> None:
        assert FAILURE_EMPTY_ANSWER in self._classify(answer="   ")

    def test_failures_are_not_mutually_exclusive(self) -> None:
        """One answer can fail several ways; collapsing them loses signal."""
        failures = self._classify(
            citation_scores={"citation_recall": 0.0, "citation_precision": 0.0},
            invalid_count=2,
            malformed=True,
            claim_coverage=0.0,
        )
        assert len(failures) >= 4

    def test_no_targets_suppresses_relevance_failures(self) -> None:
        failures = self._classify(
            has_targets=False,
            citation_scores={"citation_recall": 0.0, "citation_precision": 0.0},
        )
        assert FAILURE_MISSING_EXPECTED_CITATION not in failures
        assert FAILURE_IRRELEVANT_VALID_CITATION not in failures


# ---------------------------------------------------------------------------
# Paired comparison (spec section 43)
# ---------------------------------------------------------------------------


class TestPairedComparison:
    def test_counts_wins_ties_losses(self) -> None:
        result = paired_comparison([0.0, 0.5, 1.0], [1.0, 0.5, 0.0])
        assert result["treatment_better"] == 1
        assert result["tie"] == 1
        assert result["baseline_better"] == 1
        assert result["mean_delta"] == pytest.approx(0.0)

    def test_none_values_skipped_not_counted_as_ties(self) -> None:
        """An absent metric must never read as a tie."""
        result = paired_comparison([None, 0.5], [1.0, None])
        assert result["skipped"] == 2
        assert result["compared"] == 0
        assert result["mean_delta"] is None

    def test_all_wins(self) -> None:
        result = paired_comparison([0.0, 0.0], [1.0, 1.0])
        assert result["treatment_better"] == 2
        assert result["mean_delta"] == pytest.approx(1.0)

    def test_empty_input(self) -> None:
        result = paired_comparison([], [])
        assert result["compared"] == 0
        assert result["mean_delta"] is None
