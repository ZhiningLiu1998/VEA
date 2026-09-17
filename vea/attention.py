"""Attention extraction and the attention statistics behind the paper's analysis."""

from __future__ import annotations

import numpy as np
import torch

from vea.layout import PromptLayout

__all__ = [
    "extract_prompt_attention",
    "layer_span_indices",
    "evidence_scores",
    "section_attention_stats",
]


@torch.no_grad()
def extract_prompt_attention(model, inputs) -> np.ndarray:
    """Head-averaged attention paid by the last prompt token to the whole prompt.

    The last prompt position is the one that produces the first answer token, so
    its attention row is what determines the answer.  Heads are averaged within
    each layer, following the paper's ``a^(l) = (1/H) sum_h a^(l,h)``.

    The model must be loaded with ``attn_implementation="eager"``; the fused
    kernels never materialise the attention matrix and silently return ``None``.

    Args:
        model: a loaded image-text-to-text model.
        inputs: processor output, already moved to the model's device.

    Returns:
        ``(n_layers, n_prompt_tokens)`` float32 attention, NaNs replaced by 0.

    Raises:
        RuntimeError: if the model did not return attention weights.
    """
    outputs = model(**inputs, output_attentions=True, use_cache=False)
    if not getattr(outputs, "attentions", None):
        raise RuntimeError(
            "the model returned no attention weights; load it with "
            'attn_implementation="eager"'
        )
    attention = np.stack(
        [layer[0, :, -1, :].mean(dim=0).float().cpu().numpy() for layer in outputs.attentions]
    )
    # Some layers emit NaN for fully-masked positions; treat those as no attention.
    return np.nan_to_num(attention, nan=0.0).astype(np.float32)


def layer_span_indices(n_layers: int, span: tuple[float, float]) -> list[int]:
    """Resolve a relative depth range to concrete layer indices.

    Args:
        n_layers: total number of layers.
        span: ``(low, high)`` fractions of depth in ``[0, 1]``, e.g. ``(0.5, 1.0)``
            for the deeper half.

    Returns:
        The layer indices in the span; never empty.
    """
    low, high = span
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(f"span must satisfy 0 <= low < high <= 1, got {span}")
    start = int(low * n_layers)
    end = int(high * n_layers)
    return list(range(start, max(end, start + 1)))


def evidence_scores(
    attention: np.ndarray, layout: PromptLayout, layers: list[int]
) -> np.ndarray:
    """Eq. (1): per-patch evidence score averaged over the selected layers.

    Args:
        attention: ``(n_layers, n_prompt_tokens)`` from :func:`extract_prompt_attention`.
        layout: the prompt layout, providing the image token span.
        layers: layer indices to average, e.g. the profiled visual-grounding layers.

    Returns:
        ``(n_patches,)`` float32 scores in row-major patch order.
    """
    if not layers:
        raise ValueError("layers must not be empty")
    n_layers = attention.shape[0]
    out_of_range = [i for i in layers if not 0 <= i < n_layers]
    if out_of_range:
        raise IndexError(f"layer indices {out_of_range} outside [0, {n_layers})")

    start, end = layout.image_span
    return attention[layers, start:end].mean(axis=0).astype(np.float32)


def section_attention_stats(
    attention: np.ndarray, layout: PromptLayout, evidence_patch_mask: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Per-layer attention mass per token, for each section of the prompt.

    Two views are returned for every section:

    * ``<section>_mean`` -- raw mean attention per token in that section.
    * ``<section>_mean_norm`` -- Relative Attention Per Token (RAPT), the same
      quantity divided by the prompt-wide mean. RAPT is scale-free, so it is
      comparable across layers and models: 1.0 means the section receives its
      proportional share, and 0.6 means 60% of it. This is the quantity plotted
      in the paper's layer-dynamics figures.

    Args:
        attention: ``(n_layers, n_prompt_tokens)`` from :func:`extract_prompt_attention`.
        layout: the prompt layout.
        evidence_patch_mask: ``(n_patches,)`` bool labels. When given, the image
            section is additionally split into evidence and non-evidence tokens,
            which is what separates "seeing" from "believing".

    Returns:
        A dict of ``(n_layers,)`` arrays keyed as described above. Sections that
        are empty for this sample are reported as NaN.
    """
    image_start, image_end = layout.image_span
    question_start, question_end = layout.question_span

    sections: dict[str, np.ndarray] = {
        "input": attention,
        "image": attention[:, image_start:image_end],
        "quest": attention[:, question_start:question_end],
    }

    if evidence_patch_mask is not None:
        image_attention = sections["image"]
        mask = np.asarray(evidence_patch_mask, dtype=bool)
        if mask.shape[0] != image_attention.shape[1]:
            raise ValueError(
                f"evidence mask has {mask.shape[0]} patches but the prompt has "
                f"{image_attention.shape[1]} image tokens"
            )
        sections["image_evd"] = image_attention[:, mask]
        sections["image_nonevd"] = image_attention[:, ~mask]

    def per_token_mean(block: np.ndarray) -> np.ndarray:
        if block.shape[1] == 0:
            return np.full(block.shape[0], np.nan, dtype=np.float32)
        return block.mean(axis=1).astype(np.float32)

    input_mean = per_token_mean(sections["input"])
    stats: dict[str, np.ndarray] = {}
    for name, block in sections.items():
        mean = per_token_mean(block)
        stats[f"{name}_mean"] = mean
        if name != "input":
            with np.errstate(divide="ignore", invalid="ignore"):
                stats[f"{name}_mean_norm"] = (mean / input_mean).astype(np.float32)
    return stats
