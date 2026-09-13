"""Unit tests for mrta.eval.generation_metrics.

Pure functions over canonical evidence and strings — no model, no network.
"""

from __future__ import annotations

import pytest

from mrta.eval.generation_metrics import (
    citation_precision_recall,
    citation_validity,
    count_tokens,
    evidence_coverage,
    lexical_support_score,
    size_metrics,
    unsupported_claim_fraction,
    unsupported_numeric_tokens,
)
from mrta.eval.types import CanonicalEvidence

DOC = "attention_is_all_you_need"


def text_ev(page: int, doc: str = DOC) -> CanonicalEvidence:
    return CanonicalEvidence(document_id=doc, page_number=page, figure_id=None)


def fig_ev(page: int, figure_id: str, doc: str = DOC) -> CanonicalEvidence:
    return CanonicalEvidence(document_id=doc, page_number=page, figure_id=figure_id)


# ---------------------------------------------------------------------------
# Citation precision / recall / F1
# ---------------------------------------------------------------------------


class TestCitationPrecisionRecall:
    def test_exact_match(self) -> None:
        target = [text_ev(3)]
        scores = citation_precision_recall([text_ev(3)], target)
        assert scores["citation_precision"] == 1.0
        assert scores["citation_recall"] == 1.0
        assert scores["citation_f1"] == 1.0

    def test_duplicate_response_citations_do_not_inflate(self) -> None:
        """Citing one target twice is one piece of evidence."""
        scores = citation_precision_recall([text_ev(3), text_ev(3)], [text_ev(3)])
        assert scores["citation_precision"] == 1.0
        assert scores["citation_recall"] == 1.0

    def test_duplicate_citations_do_not_deflate_precision(self) -> None:
        scores = citation_precision_recall([text_ev(3), text_ev(3), text_ev(9)], [text_ev(3)])
        # deduped to {p3, p9}: one of two correct
        assert scores["citation_precision"] == pytest.approx(0.5)
        assert scores["citation_recall"] == 1.0

    def test_missing_expected_citation(self) -> None:
        scores = citation_precision_recall([text_ev(3)], [text_ev(3), text_ev(5)])
        assert scores["citation_precision"] == 1.0
        assert scores["citation_recall"] == pytest.approx(0.5)

    def test_extra_irrelevant_citation_costs_precision_not_recall(self) -> None:
        scores = citation_precision_recall([text_ev(3), text_ev(99)], [text_ev(3)])
        assert scores["citation_precision"] == pytest.approx(0.5)
        assert scores["citation_recall"] == 1.0

    def test_phantom_citation_to_other_document(self) -> None:
        scores = citation_precision_recall([text_ev(3, doc="other_paper")], [text_ev(3)])
        assert scores["citation_precision"] == 0.0
        assert scores["citation_recall"] == 0.0

    def test_empty_response_citations(self) -> None:
        """Nothing cited: nothing wrong was claimed, but nothing was recalled."""
        scores = citation_precision_recall([], [text_ev(3)])
        assert scores["citation_precision"] == 1.0
        assert scores["citation_recall"] == 0.0
        assert scores["citation_f1"] == pytest.approx(0.0)

    def test_empty_expected_evidence_with_no_citations(self) -> None:
        scores = citation_precision_recall([], [])
        assert scores["citation_precision"] == 1.0
        assert scores["citation_recall"] == 1.0

    def test_empty_expected_evidence_with_citations(self) -> None:
        scores = citation_precision_recall([text_ev(3)], [])
        assert scores["citation_precision"] == 0.0
        assert scores["citation_recall"] == 1.0

    def test_multiple_expected_targets_all_cited(self) -> None:
        targets = [text_ev(3), text_ev(5), fig_ev(4, "p4_f1")]
        scores = citation_precision_recall(list(targets), targets)
        assert scores["citation_recall"] == 1.0
        assert scores["citation_precision"] == 1.0

    def test_hybrid_text_and_figure_targets_partially_cited(self) -> None:
        targets = [text_ev(3), fig_ev(4, "p4_f1")]
        scores = citation_precision_recall([fig_ev(4, "p4_f1")], targets)
        assert scores["citation_recall"] == pytest.approx(0.5)
        assert scores["citation_precision"] == 1.0

    def test_figure_citation_does_not_satisfy_text_target_on_same_page(self) -> None:
        """Page-level collapsing is exactly what canonical identity prevents."""
        scores = citation_precision_recall([fig_ev(3, "p3_f1")], [text_ev(3)])
        assert scores["citation_precision"] == 0.0
        assert scores["citation_recall"] == 0.0

    def test_canonical_duplicate_across_modalities_counts_once(self) -> None:
        """The same figure cited via two labels is one canonical citation."""
        scores = citation_precision_recall(
            [fig_ev(4, "p4_f1"), fig_ev(4, "p4_f1")], [fig_ev(4, "p4_f1")]
        )
        assert scores["citation_precision"] == 1.0
        assert scores["citation_recall"] == 1.0

    def test_f1_is_harmonic_mean(self) -> None:
        scores = citation_precision_recall([text_ev(3), text_ev(99)], [text_ev(3), text_ev(5)])
        p, r = scores["citation_precision"], scores["citation_recall"]
        assert p == pytest.approx(0.5) and r == pytest.approx(0.5)
        assert scores["citation_f1"] == pytest.approx(2 * p * r / (p + r))


# ---------------------------------------------------------------------------
# Validity vs relevance
# ---------------------------------------------------------------------------


class TestCitationValidity:
    def test_all_valid(self) -> None:
        v = citation_validity(["[T1]", "[F1]"], [])
        assert v.validity_rate == 1.0
        assert v.hallucinated_citation_rate == 0.0

    def test_invalid_label_counted_as_hallucinated(self) -> None:
        v = citation_validity(["[T1]"], ["[F9]"])
        assert v.valid_count == 1 and v.invalid_count == 1
        assert v.validity_rate == pytest.approx(0.5)
        assert v.hallucinated_citation_rate == pytest.approx(0.5)
        assert v.unknown_labels == ["[F9]"]

    def test_no_citations_is_valid_not_hallucinated(self) -> None:
        v = citation_validity([], [])
        assert v.validity_rate == 1.0
        assert v.hallucinated_citation_rate == 0.0

    def test_duplicate_labels_counted_once(self) -> None:
        v = citation_validity(["[T1]", "[T1]"], ["[F9]", "[F9]"])
        assert v.valid_count == 1 and v.invalid_count == 1

    def test_valid_but_irrelevant_is_not_hallucination(self) -> None:
        """A supplied-but-wrong citation is a precision loss, not fabrication."""
        v = citation_validity(["[T1]"], [])
        assert v.hallucinated_citation_rate == 0.0
        scores = citation_precision_recall([text_ev(99)], [text_ev(3)])
        assert scores["citation_precision"] == 0.0

    def test_as_dict_shape(self) -> None:
        d = citation_validity(["[T1]"], ["[F9]"]).as_dict()
        for key in (
            "valid_citation_count",
            "invalid_citation_count",
            "citation_validity_rate",
            "hallucinated_citation_rate",
            "unknown_citation_labels",
        ):
            assert key in d


# ---------------------------------------------------------------------------
# Evidence coverage
# ---------------------------------------------------------------------------


class TestEvidenceCoverage:
    def test_hybrid_both_targets_covered(self) -> None:
        targets = [text_ev(3), fig_ev(4, "p4_f1")]
        c = evidence_coverage(targets, targets)
        assert c["text_target_covered"] is True
        assert c["figure_target_covered"] is True
        assert c["both_targets_covered"] is True
        assert c["all_targets_covered"] is True

    def test_hybrid_only_text_covered(self) -> None:
        targets = [text_ev(3), fig_ev(4, "p4_f1")]
        c = evidence_coverage([text_ev(3)], targets)
        assert c["text_target_covered"] is True
        assert c["figure_target_covered"] is False
        assert c["both_targets_covered"] is False
        assert c["any_target_covered"] is True
        assert c["all_targets_covered"] is False

    def test_both_targets_is_none_for_text_only_query(self) -> None:
        """N/A rather than a silent success."""
        c = evidence_coverage([text_ev(3)], [text_ev(3)])
        assert c["both_targets_covered"] is None
        assert c["figure_target_covered"] is None

    def test_no_citations_covers_nothing(self) -> None:
        c = evidence_coverage([], [text_ev(3)])
        assert c["any_target_covered"] is False
        assert c["text_target_covered"] is False

    def test_no_targets_reports_none(self) -> None:
        c = evidence_coverage([text_ev(3)], [])
        assert c["any_target_covered"] is None
        assert c["all_targets_covered"] is None


# ---------------------------------------------------------------------------
# Grounding proxies
# ---------------------------------------------------------------------------


class TestLexicalSupportProxy:
    def test_fully_supported_answer(self) -> None:
        ctx = "The encoder is composed of a stack of six identical layers."
        assert lexical_support_score("The encoder has identical layers.", ctx) == 1.0

    def test_unsupported_answer_scores_low(self) -> None:
        score = lexical_support_score(
            "Quantum entanglement governs photosynthesis efficiency.",
            "The encoder is composed of a stack of identical layers.",
        )
        assert score < 0.3

    def test_empty_answer_is_vacuously_supported(self) -> None:
        assert lexical_support_score("", "some context") == 1.0

    def test_empty_context_supports_nothing(self) -> None:
        assert lexical_support_score("A substantive claim here.", "") == 0.0

    def test_punctuation_and_case_normalised(self) -> None:
        assert lexical_support_score("ENCODER, layers!", "the encoder has layers") == 1.0

    def test_paraphrase_limitation_is_documented_behaviour(self) -> None:
        """A correct paraphrase scores low — the known weakness of a lexical proxy."""
        score = lexical_support_score(
            "Six stacked blocks form the encoding module.",
            "The encoder is composed of a stack of N = 6 identical layers.",
        )
        assert score < 0.5  # semantically right, lexically unsupported

    def test_numeric_tokens_are_retained(self) -> None:
        """Numbers survive the token-length floor, so a wrong figure costs score."""
        context = "hidden size 512 units"
        # every non-numeric token supported -> the number decides the score
        assert lexical_support_score("hidden size 512", context) == 1.0
        assert lexical_support_score("hidden size 999", context) == pytest.approx(2 / 3)


class TestUnsupportedClaimFraction:
    def test_all_sentences_supported(self) -> None:
        ctx = "The encoder is composed of a stack of identical layers with attention."
        result = unsupported_claim_fraction("The encoder has identical layers.", ctx)
        assert result["supported_claim_fraction"] == 1.0
        assert result["unsupported_claim_fraction"] == 0.0

    def test_mixed_support(self) -> None:
        ctx = "The encoder is composed of identical layers."
        answer = "The encoder has identical layers. Photosynthesis drives quantum tunnelling."
        result = unsupported_claim_fraction(answer, ctx)
        assert result["claim_count"] == 2
        assert result["unsupported_claim_fraction"] == pytest.approx(0.5)
        assert result["unsupported_claims"]

    def test_empty_answer(self) -> None:
        result = unsupported_claim_fraction("", "context")
        assert result["supported_claim_fraction"] == 1.0
        assert result["claim_count"] == 0

    def test_empty_context_makes_claims_unsupported(self) -> None:
        result = unsupported_claim_fraction("A substantive factual claim.", "")
        assert result["unsupported_claim_fraction"] == 1.0

    def test_threshold_is_stricter_than_any_token_overlap(self) -> None:
        """One incidental shared word must not mark a sentence supported."""
        ctx = "The encoder is composed of identical layers."
        answer = "Encoder aside, quantum tunnelling explains photosynthetic yield."
        result = unsupported_claim_fraction(answer, ctx)
        assert result["unsupported_claim_fraction"] == 1.0

    def test_unsupported_claims_list_is_capped(self) -> None:
        answer = " ".join(f"Fabricated statement number {i} entirely." for i in range(20))
        result = unsupported_claim_fraction(answer, "unrelated context")
        assert len(result["unsupported_claims"]) <= 5


class TestUnsupportedNumericTokens:
    def test_detects_fabricated_number(self) -> None:
        assert "999" in unsupported_numeric_tokens("The result was 999.", "value was 512")

    def test_supported_number_not_flagged(self) -> None:
        assert unsupported_numeric_tokens("The result was 512.", "value was 512") == []

    def test_no_numbers_returns_empty(self) -> None:
        assert unsupported_numeric_tokens("No digits here.", "context") == []


# ---------------------------------------------------------------------------
# Size metrics
# ---------------------------------------------------------------------------


class TestSizeMetrics:
    def test_token_count_reports_its_method(self) -> None:
        count, method = count_tokens("The encoder has identical layers.")
        assert count > 0
        assert method in ("tiktoken:cl100k_base", "approx:chars/4")

    def test_empty_text(self) -> None:
        assert count_tokens("") == (0, "empty")

    def test_size_metrics_shape(self) -> None:
        m = size_metrics("some context text", "an answer")
        assert m["context_char_count"] == len("some context text")
        assert m["answer_char_count"] == len("an answer")
        assert m["context_token_count"] > 0
        assert m["answer_token_count"] > 0
        assert "token_count_method" in m

    def test_longer_context_counts_more_tokens(self) -> None:
        short = size_metrics("short", "a")["context_token_count"]
        long = size_metrics("a much longer piece of context text here", "a")["context_token_count"]
        assert long > short
