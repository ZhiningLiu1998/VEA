"""Tests for the attention statistics behind the paper's analysis."""

from __future__ import annotations

import numpy as np
import pytest

from vea.attention import evidence_scores, layer_span_indices, section_attention_stats
from vea.layout import PromptLayout


def make_layout(
    image_span=(2, 6), question_span=(6, 9), n_input_tokens=10, grid_size=(2, 2)
) -> PromptLayout:
    """A minimal layout: 4 image tokens on a 2x2 grid of 1x1-pixel patches."""
    cols, rows = grid_size
    return PromptLayout(
        image_span=image_span,
        question_span=question_span,
        n_input_tokens=n_input_tokens,
        image_hw=(rows, cols),
        grid_size=grid_size,
        patch_size=(1, 1),
        patch_coords=[(j, i, j + 1, i + 1) for i in range(rows) for j in range(cols)],
    )


class TestLayerSpanIndices:
    def test_full_span_covers_every_layer(self):
        assert layer_span_indices(8, (0.0, 1.0)) == list(range(8))

    def test_halves_partition_the_stack(self):
        lower = layer_span_indices(8, (0.0, 0.5))
        upper = layer_span_indices(8, (0.5, 1.0))
        assert lower == [0, 1, 2, 3]
        assert upper == [4, 5, 6, 7]
        assert set(lower) | set(upper) == set(range(8))

    def test_narrow_span_still_yields_a_layer(self):
        assert layer_span_indices(4, (0.5, 0.51)) == [2]

    @pytest.mark.parametrize("span", [(0.5, 0.5), (0.7, 0.3), (-0.1, 0.5), (0.0, 1.5)])
    def test_invalid_spans_are_rejected(self, span):
        with pytest.raises(ValueError, match="span must satisfy"):
            layer_span_indices(8, span)


class TestEvidenceScores:
    def test_averages_the_selected_layers_over_the_image_span(self):
        attention = np.zeros((4, 10), dtype=np.float32)
        attention[1, 2:6] = [1.0, 2.0, 3.0, 4.0]
        attention[3, 2:6] = [3.0, 2.0, 1.0, 0.0]
        out = evidence_scores(attention, make_layout(), layers=[1, 3])
        np.testing.assert_allclose(out, [2.0, 2.0, 2.0, 2.0])

    def test_ignores_tokens_outside_the_image_span(self):
        attention = np.zeros((2, 10), dtype=np.float32)
        attention[0, :] = 9.0          # everything
        attention[0, 2:6] = 1.0        # except the image tokens
        out = evidence_scores(attention, make_layout(), layers=[0])
        np.testing.assert_allclose(out, [1.0, 1.0, 1.0, 1.0])

    def test_result_length_matches_the_patch_count(self):
        attention = np.random.default_rng(0).random((4, 10)).astype(np.float32)
        assert evidence_scores(attention, make_layout(), [0]).shape == (4,)

    def test_empty_layer_list_is_rejected(self):
        with pytest.raises(ValueError, match="layers must not be empty"):
            evidence_scores(np.zeros((4, 10), dtype=np.float32), make_layout(), [])

    def test_out_of_range_layer_is_rejected(self):
        with pytest.raises(IndexError, match=r"outside \[0, 4\)"):
            evidence_scores(np.zeros((4, 10), dtype=np.float32), make_layout(), [0, 9])


class TestSectionAttentionStats:
    def test_rapt_is_one_when_attention_is_uniform(self):
        attention = np.full((3, 10), 0.1, dtype=np.float32)
        stats = section_attention_stats(attention, make_layout())
        np.testing.assert_allclose(stats["image_mean_norm"], 1.0)
        np.testing.assert_allclose(stats["quest_mean_norm"], 1.0)

    def test_rapt_reflects_relative_concentration(self):
        # Image tokens get 4x the attention of everything else.
        attention = np.full((1, 10), 1.0, dtype=np.float32)
        attention[0, 2:6] = 4.0
        stats = section_attention_stats(attention, make_layout())
        assert stats["image_mean_norm"][0] > 1.0
        assert stats["quest_mean_norm"][0] < 1.0

    def test_evidence_split_requires_a_mask(self):
        attention = np.random.default_rng(0).random((2, 10)).astype(np.float32)
        assert "image_evd_mean" not in section_attention_stats(attention, make_layout())

    def test_evidence_and_nonevidence_are_separated(self):
        attention = np.zeros((1, 10), dtype=np.float32)
        attention[0, 2:6] = [10.0, 10.0, 1.0, 1.0]
        mask = np.array([True, True, False, False])
        stats = section_attention_stats(attention, make_layout(), mask)
        assert stats["image_evd_mean"][0] == pytest.approx(10.0)
        assert stats["image_nonevd_mean"][0] == pytest.approx(1.0)
        # This ratio is the paper's diagnostic quantity.
        assert stats["image_evd_mean_norm"][0] > stats["image_nonevd_mean_norm"][0]

    def test_all_evidence_gives_nan_for_the_empty_complement(self):
        attention = np.full((1, 10), 1.0, dtype=np.float32)
        stats = section_attention_stats(attention, make_layout(), np.ones(4, dtype=bool))
        assert np.isnan(stats["image_nonevd_mean"][0])

    def test_mask_length_mismatch_is_rejected(self):
        attention = np.zeros((1, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="evidence mask has 3 patches"):
            section_attention_stats(attention, make_layout(), np.ones(3, dtype=bool))

    def test_one_value_per_layer(self):
        attention = np.random.default_rng(0).random((7, 10)).astype(np.float32)
        stats = section_attention_stats(attention, make_layout(), np.array([1, 1, 0, 0], bool))
        for name, values in stats.items():
            assert values.shape == (7,), name
