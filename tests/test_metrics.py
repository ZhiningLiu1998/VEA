"""Tests for the answer-correctness and attribution metrics."""

from __future__ import annotations

import numpy as np
import pytest

from vea.metrics import ATTRIBUTION_METRICS, score_answer, score_attribution


class TestScoreAnswer:
    def test_exact_answer_scores_perfectly(self):
        scores = score_answer(["nokia"], "nokia")
        assert scores["em"] == 1.0
        assert scores["f1"] == pytest.approx(1.0)

    def test_wrong_answer_scores_zero_exact_match(self):
        assert score_answer(["nokia"], "samsung")["em"] == 0.0

    def test_any_reference_may_match(self):
        # Exact match is evaluated against the whole set, so a valid alternative
        # phrasing still counts.
        assert score_answer(["nokia", "nokia phone"], "nokia phone")["em"] == 1.0

    def test_token_metrics_come_from_the_best_reference(self):
        # The second reference matches exactly, so F1 must reflect that one.
        scores = score_answer(["totally different", "nokia"], "nokia")
        assert scores["f1"] == pytest.approx(1.0)

    def test_partial_overlap_lands_between_zero_and_one(self):
        scores = score_answer(["red bicycle"], "bicycle")
        assert 0.0 < scores["f1"] < 1.0

    def test_pr_and_re_follow_the_upstream_orientation(self):
        # qa_metrics names these the opposite way round from the usual convention:
        # `pr` is the fraction of *reference* tokens recovered and `re` the
        # fraction of *predicted* tokens that are correct. Pinned so a change in
        # the dependency is caught rather than silently reinterpreting results.
        under = score_answer(["red bicycle"], "bicycle")       # missing a token
        assert under["pr"] == pytest.approx(0.5)
        assert under["re"] == pytest.approx(1.0)

        over = score_answer(["bicycle"], "red bicycle")        # an extra token
        assert over["pr"] == pytest.approx(1.0)
        assert over["re"] == pytest.approx(0.5)

    def test_all_expected_keys_are_present(self):
        scores = score_answer(["a"], "a")
        assert set(scores) == {"em", "f1m", "f1", "pr", "re"}

    def test_empty_reference_list_is_rejected(self):
        with pytest.raises(ValueError, match="references must not be empty"):
            score_answer([], "anything")

    def test_unknown_selection_metric_is_rejected(self):
        with pytest.raises(ValueError, match="select_by must be one of"):
            score_answer(["a"], "a", select_by="bleu")


class TestScoreAttribution:
    def test_perfect_ranking_scores_one(self):
        labels = np.array([1, 1, 0, 0], dtype=bool)
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        result = score_attribution(labels, scores)
        assert result["auroc"] == pytest.approx(1.0)
        assert result["ndcg"] == pytest.approx(1.0)

    def test_inverted_ranking_scores_zero_auroc(self):
        labels = np.array([1, 1, 0, 0], dtype=bool)
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        assert score_attribution(labels, scores)["auroc"] == pytest.approx(0.0)

    def test_random_ranking_lands_near_one_half(self):
        rng = np.random.default_rng(0)
        labels = rng.random(2000) > 0.8
        result = score_attribution(labels, rng.random(2000))
        assert 0.45 < result["auroc"] < 0.55

    def test_degenerate_labels_yield_nan_rather_than_raising(self):
        # A sample whose patches are all evidence (or none) leaves AUROC
        # undefined; it must be skipped in aggregation, not crash the run.
        result = score_attribution(np.ones(5, dtype=bool), np.arange(5.0))
        assert all(np.isnan(result[name]) for name in ATTRIBUTION_METRICS)

    def test_all_expected_keys_are_present(self):
        result = score_attribution(np.array([1, 0], dtype=bool), np.array([1.0, 0.0]))
        assert set(result) == set(ATTRIBUTION_METRICS)

    def test_ndcg_at_k_only_credits_the_top_k(self):
        labels = np.array([0, 0, 0, 1], dtype=bool)
        scores = np.array([0.9, 0.8, 0.7, 0.1])  # the sole positive ranks last
        assert score_attribution(labels, scores, k=2)["ndcg"] == pytest.approx(0.0)

    def test_shape_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            score_attribution(np.array([1, 0], dtype=bool), np.array([1.0]))

    def test_accepts_boolean_and_integer_labels(self):
        scores = np.array([0.9, 0.1])
        as_bool = score_attribution(np.array([True, False]), scores)
        as_int = score_attribution(np.array([1, 0]), scores)
        assert as_bool == as_int
