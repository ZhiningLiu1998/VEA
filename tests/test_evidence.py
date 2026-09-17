"""Tests for projecting evidence boxes onto the visual token grid."""

from __future__ import annotations

import numpy as np
import pytest

from vea.evidence import _connected_boxes, _reduce_to_patches


class TestReduceToPatches:
    """A 4x4 pixel mask over 2x2 patches, i.e. a 2x2 patch grid."""

    def _mask(self, filled: tuple[slice, slice]) -> np.ndarray:
        mask = np.zeros((4, 4), dtype=bool)
        mask[filled] = True
        return mask

    def test_any_rule_marks_partially_covered_patches(self):
        # One pixel of the top-left patch is covered.
        mask = self._mask((slice(0, 1), slice(0, 1)))
        out = _reduce_to_patches(mask, 2, 2, "any")
        np.testing.assert_array_equal(out, [True, False, False, False])

    def test_all_rule_requires_full_coverage(self):
        mask = self._mask((slice(0, 1), slice(0, 1)))
        assert not _reduce_to_patches(mask, 2, 2, "all").any()

        mask = self._mask((slice(0, 2), slice(0, 2)))
        np.testing.assert_array_equal(
            _reduce_to_patches(mask, 2, 2, "all"), [True, False, False, False]
        )

    def test_half_rule_needs_half_the_pixels(self):
        # 2 of the patch's 4 pixels -> exactly half, which counts.
        mask = self._mask((slice(0, 2), slice(0, 1)))
        np.testing.assert_array_equal(
            _reduce_to_patches(mask, 2, 2, "half"), [True, False, False, False]
        )
        # 1 of 4 -> below half.
        mask = self._mask((slice(0, 1), slice(0, 1)))
        assert not _reduce_to_patches(mask, 2, 2, "half").any()

    def test_rules_are_ordered_by_strictness(self):
        rng = np.random.default_rng(0)
        mask = rng.random((8, 8)) > 0.5
        counts = {
            rule: _reduce_to_patches(mask, 2, 2, rule).sum() for rule in ("any", "half", "all")
        }
        assert counts["any"] >= counts["half"] >= counts["all"]

    def test_output_is_row_major(self):
        # Only the bottom-right patch is covered -> last entry in row-major order.
        mask = self._mask((slice(2, 4), slice(2, 4)))
        out = _reduce_to_patches(mask, 2, 2, "any")
        np.testing.assert_array_equal(out, [False, False, False, True])

    def test_unknown_rule_is_rejected(self):
        with pytest.raises(ValueError, match="rule must be one of"):
            _reduce_to_patches(np.zeros((2, 2), dtype=bool), 2, 2, "most")


class TestConnectedBoxes:
    def test_single_region_yields_its_bounding_box(self):
        mask = np.zeros((6, 6), dtype=bool)
        mask[1:3, 2:5] = True
        assert _connected_boxes(mask) == [(2, 1, 5, 3)]

    def test_disjoint_regions_are_reported_separately(self):
        mask = np.zeros((6, 6), dtype=bool)
        mask[0:2, 0:2] = True
        mask[4:6, 4:6] = True
        assert len(_connected_boxes(mask)) == 2

    def test_empty_mask_yields_nothing(self):
        assert _connected_boxes(np.zeros((4, 4), dtype=bool)) == []


class TestAdapterBoxProjection:
    """Geometry checks that need no model weights, only a stub processor."""

    def test_qwen_scales_boxes_with_the_image(self):
        from types import SimpleNamespace

        from PIL import Image

        from vea.adapters import Qwen2_5VLAdapter

        processor = SimpleNamespace(
            image_processor=SimpleNamespace(
                patch_size=14, merge_size=2, min_pixels=4 * 28 * 28,
                max_pixels=64 * 28 * 28, do_convert_rgb=True,
            )
        )
        adapter = Qwen2_5VLAdapter()
        image = Image.new("RGB", (280, 280))
        height, width = adapter.input_size(image, processor)

        boxes, oob = adapter.project_boxes(image, processor, [[0, 0, 140, 140]])
        # A box covering the top-left quarter must still cover a quarter.
        assert boxes[0][2] == pytest.approx(width / 2, abs=1)
        assert boxes[0][3] == pytest.approx(height / 2, abs=1)
        assert oob == [False]

    def test_llava_flags_boxes_cropped_away(self):
        from types import SimpleNamespace

        from PIL import Image

        from vea.adapters import LlavaAdapter

        processor = SimpleNamespace(
            patch_size=14,
            image_processor=SimpleNamespace(
                do_resize=True, size={"shortest_edge": 336},
                do_center_crop=True, crop_size={"height": 336, "width": 336},
            ),
        )
        adapter = LlavaAdapter()
        # A wide image: the centre crop discards the left and right margins.
        image = Image.new("RGB", (1000, 336))
        _, oob = adapter.project_boxes(image, processor, [[0, 0, 5, 5], [400, 100, 600, 300]])
        assert oob == [True, False]

    def test_grid_and_patch_coords_are_consistent(self):
        from types import SimpleNamespace

        from PIL import Image

        from vea.adapters import LlavaAdapter

        processor = SimpleNamespace(
            patch_size=14,
            image_processor=SimpleNamespace(
                do_resize=True, size={"shortest_edge": 336},
                do_center_crop=True, crop_size={"height": 336, "width": 336},
            ),
        )
        adapter = LlavaAdapter()
        image = Image.new("RGB", (500, 400))
        cols, rows = adapter.grid_size(image, processor)
        coords = adapter.patch_coords(image, processor)
        assert (cols, rows) == (24, 24)
        assert len(coords) == cols * rows
        # Last patch must end exactly at the image corner.
        assert coords[-1] == (336 - 14, 336 - 14, 336, 336)
