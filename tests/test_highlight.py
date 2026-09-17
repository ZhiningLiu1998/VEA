"""Tests for the evidence-map construction (Vea Steps C-E)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from vea.highlight import (
    build_evidence_map,
    denoise,
    highlight,
    normalize,
    patch_scores_to_pixel_map,
    smooth,
)


class TestDenoise:
    def test_isolated_spike_is_replaced_by_neighbourhood_mean(self):
        grid = np.full((3, 3), 1.0, dtype=np.float32)
        grid[1, 1] = 1000.0  # exceeds every neighbour by far more than lam
        out = denoise(grid, lam=10.0)
        assert out[1, 1] == pytest.approx(1.0)

    def test_coherent_cluster_is_preserved(self):
        # A 2x2 block of high values: each member has a high neighbour, so no
        # member exceeds its neighbourhood max by a factor of lam.
        grid = np.ones((4, 4), dtype=np.float32)
        grid[1:3, 1:3] = 100.0
        out = denoise(grid, lam=10.0)
        np.testing.assert_allclose(out, grid)

    def test_spike_below_threshold_is_kept(self):
        grid = np.ones((3, 3), dtype=np.float32)
        grid[1, 1] = 5.0  # only 5x the neighbours, below lam=10
        out = denoise(grid, lam=10.0)
        assert out[1, 1] == pytest.approx(5.0)

    def test_input_is_not_mutated(self):
        grid = np.ones((3, 3), dtype=np.float32)
        grid[1, 1] = 1000.0
        original = grid.copy()
        denoise(grid)
        np.testing.assert_array_equal(grid, original)

    def test_border_values_are_not_zeroed(self):
        # Border patches must survive: evidence often sits at an image edge.
        grid = np.ones((4, 4), dtype=np.float32)
        out = denoise(grid, lam=10.0)
        assert out[0, :].min() > 0
        assert out[:, 0].min() > 0
        assert out[-1, :].min() > 0

    def test_neighbourhood_excludes_centre(self):
        # With the centre included, a lone spike would be its own max and would
        # never trigger; excluding it is what makes detection possible.
        grid = np.zeros((3, 3), dtype=np.float32)
        grid[1, 1] = 1.0
        out = denoise(grid, lam=10.0)
        assert out[1, 1] == pytest.approx(0.0)

    def test_grid_smaller_than_window_is_returned_unchanged(self):
        grid = np.array([[1.0, 2.0]], dtype=np.float32)
        np.testing.assert_allclose(denoise(grid), grid)

    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError, match="2D patch grid"):
            denoise(np.zeros(9, dtype=np.float32))


class TestSmooth:
    def test_relative_sigma_scales_with_shorter_side(self):
        # sigma <= 1 is a fraction of the shorter side, so a wide image and a
        # square one with the same shorter side get the same absolute bandwidth.
        impulse_wide = np.zeros((20, 60), dtype=np.float32)
        impulse_wide[10, 30] = 1.0
        impulse_square = np.zeros((20, 20), dtype=np.float32)
        impulse_square[10, 10] = 1.0
        assert smooth(impulse_wide, 0.25)[10, 30] == pytest.approx(
            smooth(impulse_square, 0.25)[10, 10], rel=1e-3
        )

    def test_zero_sigma_is_a_no_op(self):
        grid = np.random.default_rng(0).random((8, 8)).astype(np.float32)
        np.testing.assert_allclose(smooth(grid, 0.0), grid)

    def test_blurring_spreads_mass_outward(self):
        impulse = np.zeros((21, 21), dtype=np.float32)
        impulse[10, 10] = 1.0
        out = smooth(impulse, 2.0)
        assert out[10, 10] < 1.0
        assert out[10, 12] > 0.0

    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError, match="2D pixel map"):
            smooth(np.zeros((2, 2, 2), dtype=np.float32), 0.5)


class TestNormalize:
    def test_rescales_to_unit_range(self):
        out = normalize(np.array([[2.0, 4.0], [6.0, 10.0]], dtype=np.float32))
        assert out.min() == pytest.approx(0.0)
        assert out.max() == pytest.approx(1.0)

    def test_constant_map_becomes_zero(self):
        out = normalize(np.full((3, 3), 7.0, dtype=np.float32))
        np.testing.assert_allclose(out, 0.0)

    def test_preserves_ordering(self):
        values = np.array([[1.0, 5.0, 3.0]], dtype=np.float32)
        out = normalize(values)
        assert out[0, 0] < out[0, 2] < out[0, 1]


class TestPatchScoresToPixelMap:
    def test_each_patch_paints_its_own_box(self):
        coords = [(0, 0, 2, 2), (2, 0, 4, 2), (0, 2, 2, 4), (2, 2, 4, 4)]
        out = patch_scores_to_pixel_map(np.array([1.0, 2.0, 3.0, 4.0]), coords, (4, 4))
        assert out[0, 0] == 1.0
        assert out[0, 3] == 2.0
        assert out[3, 0] == 3.0
        assert out[3, 3] == 4.0

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="scores but"):
            patch_scores_to_pixel_map(np.array([1.0]), [(0, 0, 1, 1), (1, 0, 2, 1)], (1, 2))


class TestHighlight:
    def test_full_evidence_leaves_pixels_untouched(self):
        image = Image.new("RGB", (4, 4), (200, 100, 50))
        out = highlight(image, np.ones((4, 4), dtype=np.float32), alpha=0.5)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(image))

    def test_zero_evidence_dims_to_alpha(self):
        image = Image.new("RGB", (4, 4), (200, 100, 50))
        out = highlight(image, np.zeros((4, 4), dtype=np.float32), alpha=0.5)
        np.testing.assert_allclose(np.asarray(out)[0, 0], [100, 50, 25], atol=1)

    def test_alpha_one_is_a_no_op_everywhere(self):
        image = Image.new("RGB", (4, 4), (200, 100, 50))
        out = highlight(image, np.zeros((4, 4), dtype=np.float32), alpha=1.0)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(image))

    def test_shape_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="does not match image"):
            highlight(Image.new("RGB", (4, 4)), np.zeros((2, 2), dtype=np.float32), 0.5)

    def test_alpha_out_of_range_is_rejected(self):
        with pytest.raises(ValueError, match=r"alpha must be in \[0, 1\]"):
            highlight(Image.new("RGB", (2, 2)), np.zeros((2, 2), dtype=np.float32), 1.5)


class TestBuildEvidenceMap:
    #: A 4x4 grid of 2x2-pixel patches covering an 8x8 image.
    GRID = (4, 4)
    COORDS = [
        (j * 2, i * 2, (j + 1) * 2, (i + 1) * 2) for i in range(4) for j in range(4)
    ]

    def _build(self, scores, **kwargs):
        return build_evidence_map(
            np.asarray(scores, dtype=np.float32),
            grid_size=self.GRID,
            patch_coords=self.COORDS,
            image_hw=(8, 8),
            **kwargs,
        )

    def test_output_is_normalized_and_image_shaped(self):
        rng = np.random.default_rng(0)
        out = self._build(rng.random(16))
        assert out.shape == (8, 8)
        assert 0.0 <= out.min() and out.max() <= 1.0

    def test_row_major_ordering_matches_patch_coords(self):
        # Only the last patch is hot; with smoothing off, only its box lights up.
        scores = np.zeros(16, dtype=np.float32)
        scores[-1] = 1.0
        out = self._build(scores, apply_denoise=False, apply_smooth=False)
        assert out[7, 7] == pytest.approx(1.0)
        assert out[0, 0] == pytest.approx(0.0)

    def test_denoise_suppresses_a_lone_spike(self):
        scores = np.ones(16, dtype=np.float32)
        scores[5] = 1000.0
        with_denoise = self._build(scores, apply_smooth=False)
        without = self._build(scores, apply_denoise=False, apply_smooth=False)
        # Denoising flattens the map, so the spike no longer dominates.
        assert with_denoise.max() == with_denoise.min()
        assert without.max() > without.min()

    def test_ablation_flags_change_the_result(self):
        rng = np.random.default_rng(1)
        scores = rng.random(16).astype(np.float32)
        full = self._build(scores)
        no_smooth = self._build(scores, apply_smooth=False)
        assert not np.allclose(full, no_smooth)
