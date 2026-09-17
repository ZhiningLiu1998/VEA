"""Plotting helpers for the qualitative figures."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from PIL import Image

__all__ = ["draw_boxes", "show_image", "show_attention_overlay", "show_layer_grid"]


def draw_boxes(ax, boxes: Sequence[Sequence[float]], color: str = "red", linewidth: float = 1.5):
    """Outline boxes on an existing axis."""
    for x0, y0, x1, y1 in boxes:
        ax.add_patch(
            patches.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                linewidth=linewidth,
                edgecolor=color,
                facecolor="none",
            )
        )
    return ax


def show_image(
    image: Image.Image,
    boxes: Sequence[Sequence[float]] = (),
    patch_boxes: Sequence[Sequence[float]] = (),
    title: str = "",
    ax=None,
):
    """Show an image with annotated boxes and their patch-grid quantization.

    ``boxes`` are drawn in red and ``patch_boxes`` in blue, so the gap between
    them shows how much precision is lost when evidence is snapped to the token
    grid the model can actually address.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(image)
    draw_boxes(ax, boxes, color="red")
    draw_boxes(ax, patch_boxes, color="blue", linewidth=1.0)
    ax.set(xticks=[], yticks=[], title=title)
    return ax


def show_attention_overlay(
    image: Image.Image,
    evidence_map: np.ndarray,
    alpha: float = 0.5,
    cmap: str = "coolwarm",
    title: str = "",
    ax=None,
):
    """Overlay a normalized evidence map on an image as a heatmap."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(image)
    ax.imshow(evidence_map, cmap=cmap, alpha=alpha, vmin=0.0, vmax=1.0)
    ax.set(xticks=[], yticks=[], title=title)
    return ax


def show_layer_grid(
    image: Image.Image,
    attention: np.ndarray,
    layout,
    layers: Sequence[int],
    boxes: Sequence[Sequence[float]] = (),
    ncols: int = 6,
    cmap: str = "coolwarm",
):
    """Per-layer attention heatmaps over one image.

    This is the view behind the paper's layer-dynamics figure: shallow layers look
    near-uniform, middle layers unstructured, and deep layers sparse but locked
    onto the annotated evidence.

    Args:
        image: the model input image.
        attention: ``(n_layers, n_prompt_tokens)`` attention.
        layout: the sample's :class:`~vea.layout.PromptLayout`.
        layers: which layers to draw.
        boxes: evidence boxes to outline on every panel.
        ncols: panels per row.
        cmap: matplotlib colormap.

    Returns:
        The created figure.
    """
    from vea.highlight import normalize, patch_scores_to_pixel_map

    nrows = int(np.ceil(len(layers) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.6 * nrows), squeeze=False)
    start, end = layout.image_span

    # axes.flat is longer than `layers` when the grid is not exactly filled.
    for ax, layer in zip(axes.flat, layers, strict=False):
        pixel_map = patch_scores_to_pixel_map(
            attention[layer, start:end], layout.patch_coords, layout.image_hw
        )
        show_attention_overlay(
            image, normalize(pixel_map), title=f"layer {layer}", ax=ax, cmap=cmap
        )
        draw_boxes(ax, boxes, color="red")

    for ax in axes.flat[len(layers) :]:
        ax.axis("off")
    fig.tight_layout()
    return fig
