"""Per-family adapters describing how a VLM turns an image into visual tokens.

Attention-based attribution needs three model-specific facts that the
``transformers`` API does not expose uniformly:

1. which positions in ``input_ids`` hold the image tokens,
2. how those tokens tile the image, i.e. the patch grid and its pixel geometry,
3. how the model's preprocessing maps original-image coordinates (in which the
   evidence boxes are annotated) into that geometry.

Each :class:`VisionAdapter` subclass answers those three questions for one
processor family.  Adding support for a new family means adding one subclass and
registering it in :data:`ADAPTERS` -- no changes anywhere else in the codebase.

Only families used for *attribution* need an adapter.  A model that merely
answers questions on pre-augmented images (see the delegate-model setup in
:mod:`vea.pipeline`) does not.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

__all__ = [
    "VisionAdapter",
    "LlavaAdapter",
    "Qwen2_5VLAdapter",
    "Gemma3Adapter",
    "ADAPTERS",
    "get_adapter",
    "smart_resize",
]

BBox = Sequence[float]
"""An ``(x0, y0, x1, y1)`` pixel box."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _round_to(value: float, factor: int, mode: str = "round") -> int:
    """Snap ``value`` to the nearest multiple of ``factor``."""
    fn = {"round": round, "ceil": math.ceil, "floor": math.floor}[mode]
    return int(fn(value / factor)) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
    max_ratio: int = 200,
) -> tuple[int, int]:
    """Qwen2-VL dynamic-resolution resize rule.

    Picks the ``(height, width)`` closest to the original that is divisible by
    ``factor`` on both axes and whose pixel count lies in
    ``[min_pixels, max_pixels]``, preserving aspect ratio as far as possible.

    Returns:
        ``(height, width)`` of the resized image.
    """
    if max(height, width) / min(height, width) > max_ratio:
        raise ValueError(
            f"aspect ratio must be below {max_ratio}, "
            f"got {max(height, width) / min(height, width):.1f}"
        )
    h = max(factor, _round_to(height, factor))
    w = max(factor, _round_to(width, factor))
    if h * w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h = max(factor, _round_to(height / beta, factor, "floor"))
        w = max(factor, _round_to(width / beta, factor, "floor"))
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = _round_to(height * beta, factor, "ceil")
        w = _round_to(width * beta, factor, "ceil")
    return h, w


def _denormalize(pixel_values: torch.Tensor, image_processor) -> Image.Image:
    """Invert a processor's rescale/normalize step back to a uint8 RGB image."""
    if isinstance(pixel_values, (list, tuple)):
        pixel_values = pixel_values[0]
    if pixel_values.dim() == 4:
        pixel_values = pixel_values[0]

    mean = getattr(image_processor, "image_mean", None)
    std = getattr(image_processor, "image_std", None)
    if mean is not None and std is not None:
        device = pixel_values.device
        mean = torch.tensor(mean, device=device).view(3, 1, 1)
        std = torch.tensor(std, device=device).view(3, 1, 1)
        pixel_values = pixel_values * std + mean
    return to_pil_image((pixel_values.clamp(0, 1) * 255).to(torch.uint8))


def _image_token_id(processor) -> int:
    """Resolve the placeholder token id that stands in for image patches."""
    tokenizer = processor.tokenizer
    token_id = getattr(tokenizer, "image_token_id", None)
    if isinstance(token_id, int):
        return token_id
    for source in (processor, tokenizer):
        token = getattr(source, "image_token", None)
        if isinstance(token, str):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if isinstance(token_id, int) and token_id >= 0:
                return token_id
    raise RuntimeError(
        f"cannot resolve the image token id for {type(processor).__name__}; "
        "override VisionAdapter.image_token_id for this family"
    )


def _clip_box(box: BBox, width: int, height: int) -> list[int]:
    """Clamp a box to the image bounds and cast to int."""
    x0, y0, x1, y1 = box
    return [
        int(max(0, min(width - 1, x0))),
        int(max(0, min(height - 1, y0))),
        int(max(0, min(width - 1, x1))),
        int(max(0, min(height - 1, y1))),
    ]


def _is_out_of_bounds(box: BBox, width: int, height: int) -> bool:
    """Whether a box lies entirely outside the image bounds."""
    x0, y0, x1, y1 = box
    return x1 < 0 or x0 >= width or y1 < 0 or y0 >= height


# ---------------------------------------------------------------------------
# Adapter interface
# ---------------------------------------------------------------------------


class VisionAdapter(ABC):
    """Describes the image-to-visual-token mapping of one processor family."""

    #: ``image_processor._processor_class`` values handled by this adapter.
    processor_classes: tuple[str, ...] = ()

    #: Human-readable family name, used in error messages and result files.
    family: str = ""

    # -- token layout ------------------------------------------------------

    def image_token_id(self, processor) -> int:
        """The placeholder token id that image patches occupy."""
        return _image_token_id(processor)

    def locate_image_tokens(self, input_ids: torch.Tensor, processor) -> tuple[int, int]:
        """Return the ``[start, end)`` span of image tokens in ``input_ids``.

        The default implementation takes the span between the first and last
        occurrence of the image placeholder token.
        """
        positions = (input_ids == self.image_token_id(processor)).nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            raise ValueError(f"no image tokens found in input_ids for {self.family}")
        return int(positions[0]), int(positions[-1]) + 1

    # -- image geometry ----------------------------------------------------

    @abstractmethod
    def input_size(self, image: Image.Image, processor) -> tuple[int, int]:
        """Size of the image the model actually sees, as ``(height, width)``."""

    @abstractmethod
    def patch_size(self, processor) -> tuple[int, int]:
        """Pixel footprint of one visual token, as ``(patch_h, patch_w)``."""

    @abstractmethod
    def project_boxes(
        self, image: Image.Image, processor, boxes: Sequence[BBox]
    ) -> tuple[list[list[int]], list[bool]]:
        """Map boxes from original-image to model-input coordinates.

        Returns:
            ``(projected_boxes, out_of_bounds_flags)``. Projected boxes are
            clipped to the input image; the flag marks boxes that fell entirely
            outside it and whose clipped coordinates are therefore meaningless.
        """

    @abstractmethod
    def render_input_image(self, image: Image.Image, processor) -> Image.Image:
        """Reconstruct the model input image as a viewable RGB image.

        Highlighting is applied to *this* image so that the evidence map, which
        lives on the patch grid, is pixel-aligned with what the model sees.
        """

    # -- derived -----------------------------------------------------------

    def grid_size(self, image: Image.Image, processor) -> tuple[int, int]:
        """Visual token grid as ``(cols, rows)``."""
        height, width = self.input_size(image, processor)
        patch_h, patch_w = self.patch_size(processor)
        return width // patch_w, height // patch_h

    def patch_coords(
        self, image: Image.Image, processor
    ) -> list[tuple[int, int, int, int]]:
        """Pixel box of every visual token, in row-major order.

        The order matches how the vision encoders flatten their patch grid:
        left to right within a row, then top to bottom.
        """
        cols, rows = self.grid_size(image, processor)
        patch_h, patch_w = self.patch_size(processor)
        return [
            (j * patch_w, i * patch_h, (j + 1) * patch_w, (i + 1) * patch_h)
            for i in range(rows)
            for j in range(cols)
        ]


# ---------------------------------------------------------------------------
# LLaVA 1.5
# ---------------------------------------------------------------------------


class LlavaAdapter(VisionAdapter):
    """LLaVA-1.5: shortest-edge resize, centre crop, fixed square patch grid.

    Does *not* cover LLaVA-NeXT / v1.6, whose ``anyres`` scheme emits several
    tiles per image and therefore needs a different grid model.
    """

    processor_classes = ("LlavaProcessor",)
    family = "llava"

    def input_size(self, image: Image.Image, processor) -> tuple[int, int]:
        crop = processor.image_processor.crop_size
        return crop["height"], crop["width"]

    def patch_size(self, processor) -> tuple[int, int]:
        size = processor.patch_size
        return size, size

    def project_boxes(self, image, processor, boxes):
        image_processor = processor.image_processor
        orig_w, orig_h = image.size

        if image_processor.do_resize:
            scale = image_processor.size["shortest_edge"] / min(orig_w, orig_h)
        else:
            scale = 1.0
        resized_w, resized_h = orig_w * scale, orig_h * scale

        target_h, target_w = self.input_size(image, processor)
        if image_processor.do_center_crop:
            crop_x0 = (resized_w - target_w) / 2
            crop_y0 = (resized_h - target_h) / 2
        else:
            crop_x0 = crop_y0 = 0.0

        projected, out_of_bounds = [], []
        for x0, y0, x1, y1 in boxes:
            box = (
                x0 * scale - crop_x0,
                y0 * scale - crop_y0,
                x1 * scale - crop_x0,
                y1 * scale - crop_y0,
            )
            out_of_bounds.append(_is_out_of_bounds(box, target_w, target_h))
            projected.append(_clip_box(box, target_w, target_h))
        return projected, out_of_bounds

    def render_input_image(self, image, processor):
        inputs = processor(images=image, text="", return_tensors="pt")
        return _denormalize(inputs.pixel_values, processor.image_processor)


# ---------------------------------------------------------------------------
# Qwen2.5-VL
# ---------------------------------------------------------------------------


class Qwen2_5VLAdapter(VisionAdapter):
    """Qwen2.5-VL: dynamic resolution, so the patch grid varies per image."""

    processor_classes = ("Qwen2_5_VLProcessor",)
    family = "qwen2_5_vl"

    def input_size(self, image: Image.Image, processor) -> tuple[int, int]:
        image_processor = processor.image_processor
        width, height = image.size
        patch_h, _ = self.patch_size(processor)
        return smart_resize(
            height,
            width,
            factor=patch_h,
            min_pixels=image_processor.min_pixels,
            max_pixels=image_processor.max_pixels,
        )

    def patch_size(self, processor) -> tuple[int, int]:
        image_processor = processor.image_processor
        # Neighbouring patches are merged before entering the language model, so
        # one visual token covers patch_size * merge_size pixels per side.
        size = image_processor.patch_size * image_processor.merge_size
        return size, size

    def project_boxes(self, image, processor, boxes):
        orig_w, orig_h = image.size
        target_h, target_w = self.input_size(image, processor)
        scale_x, scale_y = target_w / orig_w, target_h / orig_h

        projected, out_of_bounds = [], []
        for x0, y0, x1, y1 in boxes:
            box = (x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y)
            out_of_bounds.append(_is_out_of_bounds(box, target_w, target_h))
            projected.append(_clip_box(box, target_w, target_h))
        return projected, out_of_bounds

    def render_input_image(self, image, processor):
        if processor.image_processor.do_convert_rgb and image.mode != "RGB":
            image = image.convert("RGB")
        target_h, target_w = self.input_size(image, processor)
        if (target_w, target_h) == image.size:
            return image.convert("RGB")
        return image.resize((target_w, target_h)).convert("RGB")


# ---------------------------------------------------------------------------
# Gemma 3
# ---------------------------------------------------------------------------


class Gemma3Adapter(VisionAdapter):
    """Gemma-3: fixed square input, fixed number of visual tokens.

    ``image_seq_length`` visual tokens tile a square input of side
    ``size["height"]``, so the grid side is ``sqrt(image_seq_length)``.
    """

    processor_classes = ("Gemma3Processor",)
    family = "gemma3"

    def locate_image_tokens(self, input_ids, processor):
        # The placeholder run is a fixed length, so anchor on its start rather
        # than on the last occurrence (multi-image prompts would over-extend).
        positions = (input_ids == self.image_token_id(processor)).nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            raise ValueError(f"no image tokens found in input_ids for {self.family}")
        start = int(positions[0])
        return start, start + int(processor.image_processor.image_seq_length)

    def input_size(self, image: Image.Image, processor) -> tuple[int, int]:
        size = processor.image_processor.size
        return size["height"], size["width"]

    def patch_size(self, processor) -> tuple[int, int]:
        n_tokens = int(processor.image_processor.image_seq_length)
        side = int(round(math.sqrt(n_tokens)))
        if side * side != n_tokens:
            raise ValueError(f"image_seq_length {n_tokens} is not a perfect square")
        size = processor.image_processor.size
        return size["height"] // side, size["width"] // side

    def project_boxes(self, image, processor, boxes):
        orig_w, orig_h = image.size
        target_h, target_w = self.input_size(image, processor)
        image_processor = processor.image_processor

        if image_processor.do_resize:
            # Aspect-preserving resize onto the shorter side, then centre pad to
            # the square input. Whether padding or cropping occurs depends on the
            # aspect ratio; both are handled by the signed offset below.
            scale = min(target_w, target_h) / min(orig_w, orig_h)
        else:
            scale = 1.0
        resized_w, resized_h = orig_w * scale, orig_h * scale
        offset_x = (target_w - resized_w) / 2
        offset_y = (target_h - resized_h) / 2

        projected, out_of_bounds = [], []
        for x0, y0, x1, y1 in boxes:
            box = (
                x0 * scale + offset_x,
                y0 * scale + offset_y,
                x1 * scale + offset_x,
                y1 * scale + offset_y,
            )
            out_of_bounds.append(_is_out_of_bounds(box, target_w, target_h))
            projected.append(_clip_box(box, target_w, target_h))
        return projected, out_of_bounds

    def render_input_image(self, image, processor):
        inputs = processor(images=image, text=[], return_tensors="pt")
        image_processor = processor.image_processor
        pixel_values = inputs.pixel_values
        if pixel_values.dim() == 4:
            pixel_values = pixel_values[0]

        mean = torch.tensor(image_processor.image_mean).view(3, 1, 1)
        std = torch.tensor(image_processor.image_std).view(3, 1, 1)
        pixel_values = (pixel_values * std + mean).clamp(0, 1)
        scale = 1.0 / image_processor.rescale_factor
        return to_pil_image((pixel_values * scale).to(torch.uint8))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Every adapter available for attention-based attribution.
ADAPTERS: tuple[VisionAdapter, ...] = (
    LlavaAdapter(),
    Qwen2_5VLAdapter(),
    Gemma3Adapter(),
)


def get_adapter(processor) -> VisionAdapter:
    """Return the adapter matching ``processor``.

    Raises:
        NotImplementedError: if no adapter handles this processor family.
    """
    processor_class = processor.image_processor._processor_class
    for adapter in ADAPTERS:
        if processor_class in adapter.processor_classes:
            return adapter
    supported = sorted(
        name for adapter in ADAPTERS for name in adapter.processor_classes
    )
    raise NotImplementedError(
        f"no VisionAdapter for {processor_class!r}. Attribution is supported for "
        f"{supported}. Models outside this list can still answer questions on "
        "images augmented by a delegate model (see vea.pipeline)."
    )
