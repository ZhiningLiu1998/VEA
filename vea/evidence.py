"""Ground-truth evidence labels on the visual token grid.

Datasets annotate evidence as pixel boxes on the *original* image.  Attribution
is measured on visual *tokens*, so the boxes are projected into model-input
coordinates and then reduced to one binary label per patch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy.ndimage import find_objects, label

from vea.adapters import BBox, VisionAdapter, get_adapter
from vea.layout import PromptLayout

__all__ = ["OVERLAP_RULES", "EvidenceLabels", "build_evidence_labels"]

#: How much of a patch must be covered for it to count as evidence.
#:
#: ``any``  -- the patch overlaps a box at all (the paper's rule; most inclusive)
#: ``half`` -- at least half the patch's pixels are inside a box
#: ``all``  -- the patch lies entirely inside a box (most conservative)
OVERLAP_RULES = ("any", "half", "all")


@dataclass(frozen=True)
class EvidenceLabels:
    """Ground-truth evidence at pixel and patch resolution."""

    pixel_mask: np.ndarray = field(repr=False)
    """``(H, W)`` bool mask of annotated evidence pixels, in input coordinates."""

    patch_mask: np.ndarray = field(repr=False)
    """``(n_patches,)`` bool labels in row-major patch order."""

    patch_pixel_mask: np.ndarray = field(repr=False)
    """``(H, W)`` bool mask of the patches labelled as evidence.

    This is ``patch_mask`` painted back onto the pixel canvas, i.e. the evidence
    region snapped to patch boundaries. Useful for visualising the quantization
    loss between the annotation and what the model can actually attend to.
    """

    boxes: list[list[int]]
    """Annotated boxes projected into model-input coordinates."""

    patch_boxes: list[tuple[int, int, int, int]]
    """Connected components of ``patch_pixel_mask``, as boxes."""

    out_of_bounds: list[bool]
    """Per-box flag: the box fell entirely outside the model input image."""

    n_boxes: int
    """Number of annotated boxes for this sample."""

    @property
    def n_evidence_patches(self) -> int:
        return int(self.patch_mask.sum())

    @property
    def evidence_ratio(self) -> float:
        """Fraction of visual tokens labelled as evidence."""
        return float(self.patch_mask.mean()) if self.patch_mask.size else 0.0

    @property
    def any_out_of_bounds(self) -> bool:
        return any(self.out_of_bounds)


def _reduce_to_patches(
    pixel_mask: np.ndarray, patch_h: int, patch_w: int, rule: str
) -> np.ndarray:
    """Reduce a pixel mask to one bool per patch under the given overlap rule."""
    height, width = pixel_mask.shape
    tiles = pixel_mask.reshape(height // patch_h, patch_h, width // patch_w, patch_w)
    covered = tiles.sum(axis=(1, 3))
    if rule == "any":
        return (covered > 0).flatten()
    if rule == "all":
        return (covered == patch_h * patch_w).flatten()
    if rule == "half":
        return (covered >= (patch_h * patch_w) / 2).flatten()
    raise ValueError(f"rule must be one of {OVERLAP_RULES}, got {rule!r}")


def _connected_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Bounding box of every connected ``True`` region in ``mask``."""
    labelled, _ = label(mask)
    boxes = []
    for rows, cols in (s for s in find_objects(labelled) if s is not None):
        boxes.append((cols.start, rows.start, cols.stop, rows.stop))
    return boxes


def build_evidence_labels(
    boxes: Sequence[BBox],
    image: Image.Image,
    processor,
    layout: PromptLayout,
    rule: str = "any",
    adapter: VisionAdapter | None = None,
) -> EvidenceLabels:
    """Project annotated evidence boxes onto the visual token grid.

    Args:
        boxes: ``(x0, y0, x1, y1)`` boxes in *original* image coordinates.
        image: the original image.
        processor: the model's processor.
        layout: the prompt layout, providing the patch geometry.
        rule: patch overlap rule, see :data:`OVERLAP_RULES`.
        adapter: vision adapter to use; resolved from ``processor`` if omitted.

    Returns:
        The :class:`EvidenceLabels` for this sample.
    """
    adapter = adapter or get_adapter(processor)
    projected, out_of_bounds = adapter.project_boxes(image, processor, boxes)

    height, width = layout.image_hw
    pixel_mask = np.zeros((height, width), dtype=bool)
    for (x0, y0, x1, y1), oob in zip(projected, out_of_bounds, strict=True):
        if not oob:
            pixel_mask[y0:y1, x0:x1] = True

    patch_h, patch_w = layout.patch_size
    patch_mask = _reduce_to_patches(pixel_mask, patch_h, patch_w, rule)

    cols, rows = layout.grid_size
    patch_pixel_mask = np.repeat(
        np.repeat(patch_mask.reshape(rows, cols), patch_h, axis=0), patch_w, axis=1
    )

    return EvidenceLabels(
        pixel_mask=pixel_mask,
        patch_mask=patch_mask,
        patch_pixel_mask=patch_pixel_mask,
        boxes=projected,
        patch_boxes=_connected_boxes(patch_pixel_mask),
        out_of_bounds=out_of_bounds,
        n_boxes=len(projected),
    )
