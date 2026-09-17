"""Visual evidence highlighting (Vea Steps C-E).

Turns a raw per-patch attention vector into an augmented image in which
low-evidence pixels are darkened while evidence regions keep their original
appearance.  The three stages follow Eqs. (2)-(4) of the paper:

    denoise  -> Eq. (2)  suppress isolated attention spikes
    smooth   -> Eq. (3)  Gaussian blur, removes mosaic artifacts
    highlight-> Eq. (4)  brightness attenuation of low-evidence pixels
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

__all__ = [
    "patch_scores_to_pixel_map",
    "denoise",
    "smooth",
    "normalize",
    "highlight",
    "build_evidence_map",
]


def patch_scores_to_pixel_map(
    scores: np.ndarray,
    patch_coords: Sequence[tuple[int, int, int, int]],
    image_hw: tuple[int, int],
) -> np.ndarray:
    """Paint a per-patch score vector onto a pixel-resolution canvas.

    Args:
        scores: ``(n_patches,)`` score per visual token, in row-major patch order.
        patch_coords: ``(x0, y0, x1, y1)`` pixel box of each patch, same order.
        image_hw: ``(height, width)`` of the model input image.

    Returns:
        ``(height, width)`` float32 array of per-pixel scores.
    """
    if len(scores) != len(patch_coords):
        raise ValueError(
            f"got {len(scores)} scores but {len(patch_coords)} patch coordinates"
        )
    canvas = np.zeros(image_hw, dtype=np.float32)
    for score, (x0, y0, x1, y1) in zip(scores, patch_coords, strict=True):
        canvas[y0:y1, x0:x1] = score
    return canvas


def denoise(grid: np.ndarray, lam: float = 10.0) -> np.ndarray:
    """Eq. (2): replace isolated attention spikes with their local average.

    A patch is treated as an artifact when it exceeds *every* neighbour by more
    than a factor of ``lam``; genuine evidence forms spatially coherent
    clusters, whereas encoder artifacts appear as lone outliers.  Both the
    trigger (neighbourhood max) and the replacement value (neighbourhood mean)
    are computed over the 3x3 neighbourhood *excluding* the centre patch.

    Args:
        grid: ``(h, w)`` per-patch scores.
        lam: spike threshold. Higher values suppress fewer patches.

    Returns:
        A new ``(h, w)`` float32 array; the input is not modified.
    """
    if grid.ndim != 2:
        raise ValueError(f"expected a 2D patch grid, got shape {grid.shape}")
    grid = grid.astype(np.float32, copy=True)
    if grid.shape[0] < 3 or grid.shape[1] < 3:
        return grid

    # Edge-pad so that border patches see a full 3x3 window.
    padded = np.pad(grid, pad_width=1, mode="edge")
    # Stack the 8 neighbour offsets, omitting (0, 0) which is the centre.
    h, w = grid.shape
    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    neighbours = np.stack(
        [padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] for dy, dx in offsets]
    )

    is_spike = grid > lam * neighbours.max(axis=0)
    return np.where(is_spike, neighbours.mean(axis=0), grid).astype(np.float32)


def smooth(pixel_map: np.ndarray, sigma: float) -> np.ndarray:
    """Eq. (3): isotropic Gaussian blur with a resolution-relative bandwidth.

    Args:
        pixel_map: ``(H, W)`` per-pixel scores.
        sigma: relative bandwidth in ``(0, 1]``, multiplied by the shorter image
            side to obtain the pixel-space standard deviation. Values greater
            than 1 are used directly as an absolute pixel sigma. ``0`` disables
            smoothing.

    Returns:
        The blurred ``(H, W)`` float32 array.
    """
    if pixel_map.ndim != 2:
        raise ValueError(f"expected a 2D pixel map, got shape {pixel_map.shape}")
    if sigma <= 0:
        return pixel_map.astype(np.float32, copy=True)
    if sigma <= 1:
        sigma = sigma * min(pixel_map.shape)
    return gaussian_filter(pixel_map.astype(np.float32), sigma=sigma)


def normalize(pixel_map: np.ndarray) -> np.ndarray:
    """Min-max rescale to ``[0, 1]``.

    Raw attention mass is spread over hundreds of visual tokens, so absolute
    values are tiny and vary by orders of magnitude across layers and images.
    Rescaling makes the highlight strength in Eq. (4) comparable across samples.
    A constant map is mapped to all-zeros.
    """
    lo, hi = float(pixel_map.min()), float(pixel_map.max())
    if hi <= lo:
        return np.zeros_like(pixel_map, dtype=np.float32)
    return ((pixel_map - lo) / (hi - lo)).astype(np.float32)


def highlight(image: Image.Image, evidence_map: np.ndarray, alpha: float) -> Image.Image:
    """Eq. (4): attenuate brightness in proportion to the lack of evidence.

    Each channel is scaled by ``alpha + (1 - alpha) * evidence_map``, so pixels
    with ``evidence_map == 1`` keep their original value and pixels with
    ``evidence_map == 0`` are dimmed to a fraction ``alpha`` of it.

    Args:
        image: the model input image, i.e. after the model's own resize/crop.
        evidence_map: ``(H, W)`` map in ``[0, 1]`` matching ``image`` in size.
        alpha: brightness floor in ``[0, 1]``. ``1`` is a no-op; smaller values
            highlight more aggressively.

    Returns:
        A new RGB image.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    image = image.convert("RGB")
    arr = np.asarray(image, dtype=np.float32)
    if evidence_map.shape != arr.shape[:2]:
        raise ValueError(
            f"evidence map {evidence_map.shape} does not match image {arr.shape[:2]}"
        )
    gain = (alpha + (1.0 - alpha) * evidence_map)[..., None]
    return Image.fromarray(np.clip(arr * gain, 0, 255).astype(np.uint8))


def build_evidence_map(
    patch_scores: np.ndarray,
    grid_size: tuple[int, int],
    patch_coords: Sequence[tuple[int, int, int, int]],
    image_hw: tuple[int, int],
    lam: float = 10.0,
    sigma: float = 0.5,
    apply_denoise: bool = True,
    apply_smooth: bool = True,
) -> np.ndarray:
    """Run Vea Steps C-D: raw patch scores -> normalized pixel evidence map.

    Args:
        patch_scores: ``(n_patches,)`` attention-derived score per visual token.
        grid_size: ``(cols, rows)`` of the visual token grid.
        patch_coords: pixel box of each patch, in the same order as ``patch_scores``.
        image_hw: ``(height, width)`` of the model input image.
        lam: denoising threshold, see :func:`denoise`.
        sigma: relative Gaussian bandwidth, see :func:`smooth`.
        apply_denoise: set ``False`` for the "w/o Denoise" ablation.
        apply_smooth: set ``False`` for the "w/o Smoothing" ablation.

    Returns:
        ``(height, width)`` evidence map in ``[0, 1]``.
    """
    cols, rows = grid_size
    grid = np.asarray(patch_scores, dtype=np.float32).reshape(rows, cols)
    if apply_denoise:
        grid = denoise(grid, lam=lam)
    pixel_map = patch_scores_to_pixel_map(grid.flatten(), patch_coords, image_hw)
    if apply_smooth:
        pixel_map = smooth(pixel_map, sigma=sigma)
    return normalize(pixel_map)
