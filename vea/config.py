"""Registries and configuration for the Vea experiments.

Everything a reader might want to change -- prompts, model ids, hyperparameters
-- is collected here rather than scattered across the scripts.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

__all__ = [
    "PROMPTS",
    "MODELS",
    "DATASETS",
    "METHODS",
    "ModelSpec",
    "MethodSpec",
    "VeaConfig",
    "GenerationConfig",
    "resolve_model",
    "resolve_method",
]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

#: Typographic quotes around the abstention string, as used in the paper's
#: prompts. Written as escapes to keep this source file pure ASCII.
_LQUO, _RQUO = "\u201c", "\u201d"

#: Shared opening and abstention clause. The abstention option is what lets the
#: "false rejection" failure mode in the paper's analysis be observed at all: a
#: model that cannot decline never reveals that it discarded the evidence.
_PREFIX = (
    "Directly answer the question based on the image, no explanation is needed. "
    "If the image does not contain any relevant evidence, output "
    f"{_LQUO}I cannot answer based on the given image.{_RQUO} "
)

#: The sentence that points the model at the highlighted region. This single
#: clause is the only textual difference between the Inst and Vea conditions.
_HIGHLIGHT_HINT = (
    "Only use words from the picture, especially those in the highlighted region, "
    "to answer the question. "
)

#: Prompt templates, transcribed from the paper's appendix.
#:
#: The published templates carry an ``Image: {image}`` slot. Here the image is
#: supplied structurally, as its own content block in the chat template, so that
#: slot is dropped rather than left as a dangling ``Image:`` label.
PROMPTS: dict[str, str] = {
    "qa": _PREFIX + "Question: {question}",
    "vea": _PREFIX + _HIGHLIGHT_HINT + "Question: {question}",
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """A model evaluated in the paper."""

    name: str
    """Short name used on the command line and in result files."""

    hf_id: str
    """Hugging Face repository id."""

    supports_attribution: bool
    """Whether this repo can extract per-patch attention for this model.

    Attribution requires a :class:`~vea.adapters.VisionAdapter` for the model's
    processor family. Models without one can still be *evaluated* on images
    augmented by a delegate model -- which is what the paper does for LLaVA-NeXT
    and InternVL3.5, whose eager-attention extraction exhausts an 80GB GPU.
    """


MODELS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        # Attribution-capable families (a VisionAdapter exists).
        ModelSpec("llava-1.5-7b", "llava-hf/llava-1.5-7b-hf", True),
        ModelSpec("llava-1.5-13b", "llava-hf/llava-1.5-13b-hf", True),
        ModelSpec("qwen2.5-vl-7b", "Qwen/Qwen2.5-VL-7B-Instruct", True),
        ModelSpec("qwen2.5-vl-32b", "Qwen/Qwen2.5-VL-32B-Instruct", True),
        ModelSpec("gemma-3-4b", "google/gemma-3-4b-it", True),
        ModelSpec("gemma-3-27b", "google/gemma-3-27b-it", True),
        # Evaluation-only: answer questions on delegate-augmented images.
        ModelSpec("llava-next-7b", "llava-hf/llava-v1.6-mistral-7b-hf", False),
        ModelSpec("llava-next-13b", "llava-hf/llava-v1.6-vicuna-13b-hf", False),
        ModelSpec("internvl3.5-8b", "OpenGVLab/InternVL3_5-8B-HF", False),
        ModelSpec("internvl3.5-14b", "OpenGVLab/InternVL3_5-14B-HF", False),
    )
}


def resolve_model(name_or_id: str) -> ModelSpec:
    """Look up a :class:`ModelSpec` by short name or Hugging Face id.

    Unregistered ids are accepted and assumed to be evaluation-only; whether
    attribution actually works is then decided by
    :func:`vea.adapters.get_adapter` at run time.
    """
    if name_or_id in MODELS:
        return MODELS[name_or_id]
    for spec in MODELS.values():
        if spec.hf_id == name_or_id:
            return spec
    return ModelSpec(name_or_id.split("/")[-1], name_or_id, False)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

#: VQA datasets used in the paper's main tables, all with evidence-box
#: annotations taken from the VisualCoT benchmark.
DATASETS: tuple[str, ...] = ("textvqa", "docvqa", "sroie", "infographicsvqa")


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MethodSpec:
    """One row of the paper's main comparison, or one ablation of Vea."""

    name: str
    prompt: str
    """Key into :data:`PROMPTS`."""

    augment: bool
    """Whether the image is highlighted before being shown to the model."""

    denoise: bool = True
    smooth: bool = True
    profile_layers: bool = True
    """When ``False``, average over *all* layers instead of the profiled ones.

    The paper reports this as the "w/o Profiling" ablation but does not state
    which layers replace the profiled set; averaging over the full stack is the
    natural reading and matches the ``L_0-100%`` row of the attribution table.
    """

    description: str = ""


METHODS: dict[str, MethodSpec] = {
    spec.name: spec
    for spec in (
        MethodSpec("base", "qa", False, description="Base: plain QA prompt, original image"),
        MethodSpec(
            "inst",
            "vea",
            False,
            description="Inst: Vea's prompt without the visual augmentation",
        ),
        MethodSpec("vea", "vea", True, description="Vea: full method"),
        MethodSpec(
            "vea-no-denoise", "vea", True, denoise=False, description="Vea w/o Denoise"
        ),
        MethodSpec(
            "vea-no-smooth", "vea", True, smooth=False, description="Vea w/o Smoothing"
        ),
        MethodSpec(
            "vea-no-profiling",
            "vea",
            True,
            profile_layers=False,
            description="Vea w/o Profiling",
        ),
        MethodSpec(
            "vea-no-prompt",
            "qa",
            True,
            description="Vea's augmented image with the plain QA prompt",
        ),
    )
}


def resolve_method(name: str) -> MethodSpec:
    """Look up a :class:`MethodSpec` by name."""
    try:
        return METHODS[name]
    except KeyError:
        raise KeyError(
            f"unknown method {name!r}; choose from {sorted(METHODS)}"
        ) from None


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VeaConfig:
    """Vea hyperparameters. Defaults are the values used in the paper."""

    lam: float = 10.0
    """Denoising threshold. A patch is an artifact if it exceeds every neighbour
    by more than this factor."""

    sigma: float = 0.5
    """Gaussian smoothing bandwidth, relative to the shorter image side."""

    alpha: float = 0.5
    """Brightness floor for non-evidence pixels. ``1.0`` disables highlighting."""

    layer_top_fraction: float = 0.1
    """Fraction of layers kept during visual-grounding profiling."""

    overlap_rule: str = "any"
    """Patch labelling rule, see :data:`vea.evidence.OVERLAP_RULES`."""

    max_pixels: int = 512 * 512
    """Cap on processor input resolution.

    Bounds the visual token count so that eager attention -- which materialises
    an ``n_tokens x n_tokens`` matrix per layer per head -- fits in memory.
    """


@dataclass(frozen=True)
class GenerationConfig:
    """Greedy decoding settings shared by every method, for determinism."""

    max_new_tokens: int = 50
    num_beams: int = 1
    do_sample: bool = False
    seed: int = 42


@dataclass(frozen=True)
class ProfileResult:
    """Output of visual-grounding layer profiling (Vea Step A)."""

    model: str
    n_layers: int
    layers: list[int]
    """The selected visual-grounding layers, ascending."""

    layer_auroc: list[float] = field(repr=False)
    """Mean patch-level AUROC of every layer, index-aligned to the layer stack."""

    n_samples: int = 0
    dataset: str = ""

    @staticmethod
    def _mean(values: list[float]) -> float:
        """Mean over the non-NaN entries; NaN if there are none."""
        # A layer scores NaN when every diagnostic sample was degenerate for it.
        finite = [v for v in values if not math.isnan(v)]
        return statistics.fmean(finite) if finite else float("nan")

    @property
    def mean_auroc_all(self) -> float:
        """Mean AUROC across the whole stack, the paper's reference point."""
        return self._mean(self.layer_auroc)

    @property
    def mean_auroc_selected(self) -> float:
        """Mean AUROC of the selected layers."""
        return self._mean([self.layer_auroc[i] for i in self.layers])
