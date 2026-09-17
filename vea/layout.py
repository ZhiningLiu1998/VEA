"""Token-level layout of a multimodal prompt.

Attention analysis needs to know which slice of the attention vector belongs to
the image and which belongs to the question.  :class:`PromptLayout` bundles that
bookkeeping together with the patch geometry, so downstream code never has to
reason about processor internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from PIL import Image

from vea.adapters import VisionAdapter, get_adapter

__all__ = ["PromptLayout", "build_layout", "find_question_span"]


@dataclass(frozen=True)
class PromptLayout:
    """Where the image and question live inside a tokenized prompt."""

    image_span: tuple[int, int]
    """``[start, end)`` positions of the image tokens in ``input_ids``."""

    question_span: tuple[int, int]
    """``[start, end)`` positions of the question tokens in ``input_ids``."""

    n_input_tokens: int
    """Total prompt length, i.e. the width of one attention row."""

    image_hw: tuple[int, int]
    """``(height, width)`` of the image the model actually sees."""

    grid_size: tuple[int, int]
    """Visual token grid as ``(cols, rows)``."""

    patch_size: tuple[int, int]
    """``(patch_h, patch_w)`` pixel footprint of one visual token."""

    patch_coords: list[tuple[int, int, int, int]] = field(repr=False)
    """Pixel box of each visual token, in row-major order."""

    @property
    def n_image_tokens(self) -> int:
        return self.image_span[1] - self.image_span[0]

    @property
    def n_patches(self) -> int:
        cols, rows = self.grid_size
        return cols * rows


def find_question_span(
    input_ids: torch.Tensor, processor, question: str
) -> tuple[int, int] | None:
    """Locate the question's tokens inside a chat-templated prompt.

    Chat templates wrap the question in role markers and system text, and the
    tokenizer may split it differently in isolation than in context.  Several
    plausible encodings are therefore tried, and the *last* match wins because
    the question is the final user-visible part of the prompt.

    Args:
        input_ids: ``(n_tokens,)`` prompt token ids.
        processor: the model's processor, used for its tokenizer.
        question: the raw question text.

    Returns:
        ``[start, end)`` positions, or ``None`` if no encoding could be located.
    """
    tokenizer = processor.tokenizer
    haystack = input_ids.tolist()

    # A leading space changes the first sub-token for most BPE tokenizers.
    candidates = []
    for text in (question, " " + question):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        # Tokens that decode to nothing (BOS and friends) never appear inline.
        ids = [i for i in ids if tokenizer.decode([i], skip_special_tokens=True) != ""]
        if ids and ids not in candidates:
            candidates.append(ids)

    for needle in candidates:
        for start in range(len(haystack) - len(needle), -1, -1):
            if haystack[start : start + len(needle)] == needle:
                return start, start + len(needle)
    return None


def build_layout(
    inputs, image: Image.Image, processor, question: str, adapter: VisionAdapter | None = None
) -> PromptLayout:
    """Resolve the token layout and patch geometry of a prepared prompt.

    Args:
        inputs: the processor output for this sample, containing ``input_ids``.
        image: the *original* image, before the model's preprocessing.
        processor: the model's processor.
        question: the raw question text, used to locate the question tokens.
        adapter: vision adapter to use; resolved from ``processor`` if omitted.

    Returns:
        The :class:`PromptLayout` for this sample.

    Raises:
        ValueError: if the patch grid is inconsistent with the number of image
            tokens actually present in the prompt, which means the adapter's
            geometry does not match this processor configuration.
    """
    adapter = adapter or get_adapter(processor)
    input_ids = inputs["input_ids"][0]

    image_span = adapter.locate_image_tokens(input_ids, processor)
    image_hw = adapter.input_size(image, processor)
    grid_size = adapter.grid_size(image, processor)
    patch_size = adapter.patch_size(processor)
    patch_coords = adapter.patch_coords(image, processor)

    n_image_tokens = image_span[1] - image_span[0]
    n_patches = grid_size[0] * grid_size[1]
    if n_patches != n_image_tokens:
        raise ValueError(
            f"{adapter.family}: patch grid {grid_size} implies {n_patches} visual "
            f"tokens but the prompt contains {n_image_tokens}. The adapter's image "
            f"geometry (input size {image_hw}, patch size {patch_size}) does not "
            "match this processor configuration."
        )

    # Without a question span, fall back to everything after the image: for these
    # chat templates that tail is the question plus a short generation prompt.
    question_span = find_question_span(input_ids, processor, question) or (
        image_span[1],
        len(input_ids),
    )

    return PromptLayout(
        image_span=image_span,
        question_span=question_span,
        n_input_tokens=len(input_ids),
        image_hw=image_hw,
        grid_size=grid_size,
        patch_size=patch_size,
        patch_coords=patch_coords,
    )
