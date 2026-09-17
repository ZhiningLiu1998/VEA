"""End-to-end Vea: attention attribution, image augmentation, and QA evaluation.

The one abstraction that matters here is :class:`Attributor`.  It owns the model
that *looks* at the image and produces the augmented version of it, which is not
necessarily the model that *answers* the question.  Keeping the two roles
separate is what makes the paper's delegate setup fall out for free: LLaVA-NeXT
and InternVL3.5 cannot have their attention extracted within an 80GB budget, so
Qwen2.5-VL attributes on their behalf while they still do the answering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from vea.adapters import VisionAdapter, get_adapter
from vea.attention import evidence_scores, extract_prompt_attention, section_attention_stats
from vea.config import GenerationConfig, MethodSpec, VeaConfig
from vea.data import VQASample
from vea.evidence import EvidenceLabels, build_evidence_labels
from vea.highlight import build_evidence_map, highlight
from vea.layout import PromptLayout, build_layout
from vea.metrics import score_answer, score_attribution
from vea.models import LoadedModel, build_inputs, generate_answer

__all__ = ["Attribution", "Attributor", "answer", "evaluate_sample"]


@dataclass
class Attribution:
    """Everything one attribution forward pass produces for one sample."""

    layout: PromptLayout
    labels: EvidenceLabels
    attention: np.ndarray = field(repr=False)
    """``(n_layers, n_prompt_tokens)`` head-averaged attention from the last token."""

    input_image: Image.Image = field(repr=False)
    """The model input image, at the resolution the patch grid refers to."""

    @property
    def n_layers(self) -> int:
        return int(self.attention.shape[0])


class Attributor:
    """Locates visual evidence with one model's attention and highlights it.

    Args:
        loaded: the model used for attribution. Must be loaded with eager
            attention and belong to a family that has a
            :class:`~vea.adapters.VisionAdapter`.
        layers: the profiled visual-grounding layers. Defaults to the whole
            stack, which is the "w/o Profiling" setting; run
            ``scripts/profile_layers.py`` to obtain the profiled set.
        config: Vea hyperparameters.
    """

    def __init__(
        self,
        loaded: LoadedModel,
        layers: list[int] | None = None,
        config: VeaConfig | None = None,
    ):
        self.loaded = loaded
        self.config = config or VeaConfig()
        self.adapter: VisionAdapter = get_adapter(loaded.processor)
        self._layers = list(layers) if layers else None

    @property
    def profiled_layers(self) -> list[int]:
        """The visual-grounding layers, falling back to the full stack."""
        return self._layers if self._layers is not None else list(range(self.loaded.n_layers))

    def layers_for(self, use_profiled: bool) -> list[int]:
        """Select profiled layers, or the full stack for the ablation."""
        return self.profiled_layers if use_profiled else list(range(self.loaded.n_layers))

    def attend(self, sample: VQASample, prompt_template: str) -> Attribution:
        """Run one forward pass and collect attention, layout and evidence labels.

        The prompt used here is the plain QA prompt: attribution must observe
        where the model looks when asked the question normally, before any
        highlighting exists to point at.
        """
        image = sample.load_image()
        processor = self.loaded.processor

        inputs = build_inputs(processor, image, sample.question, prompt_template)
        layout = build_layout(inputs, image, processor, sample.question, self.adapter)
        labels = build_evidence_labels(
            sample.boxes, image, processor, layout, self.config.overlap_rule, self.adapter
        )
        attention = extract_prompt_attention(self.loaded.model, inputs.to(self.loaded.device))

        return Attribution(
            layout=layout,
            labels=labels,
            attention=attention,
            input_image=self.adapter.render_input_image(image, processor),
        )

    def evidence_map(
        self,
        attribution: Attribution,
        denoise: bool = True,
        smooth: bool = True,
        use_profiled_layers: bool = True,
    ) -> np.ndarray:
        """Vea Steps B-D: attention -> normalized pixel evidence map in ``[0, 1]``."""
        layers = self.layers_for(use_profiled_layers)
        scores = evidence_scores(attribution.attention, attribution.layout, layers)
        return build_evidence_map(
            scores,
            grid_size=attribution.layout.grid_size,
            patch_coords=attribution.layout.patch_coords,
            image_hw=attribution.layout.image_hw,
            lam=self.config.lam,
            sigma=self.config.sigma,
            apply_denoise=denoise,
            apply_smooth=smooth,
        )

    def augment(
        self,
        attribution: Attribution,
        denoise: bool = True,
        smooth: bool = True,
        use_profiled_layers: bool = True,
    ) -> Image.Image:
        """Vea Step E: return the highlighted image to show the answering model.

        Highlighting is applied to the *rendered model input* rather than the
        original image, so the evidence map is pixel-aligned with the patch grid
        it was derived from. Feeding that image back through a processor is a
        no-op resize, since it is already at the model's input resolution.
        """
        evidence_map = self.evidence_map(
            attribution, denoise=denoise, smooth=smooth, use_profiled_layers=use_profiled_layers
        )
        return highlight(attribution.input_image, evidence_map, alpha=self.config.alpha)

    def patch_scores(
        self, attribution: Attribution, layers: list[int], denoise: bool = True
    ) -> np.ndarray:
        """Per-patch evidence scores, for attribution scoring on the patch grid."""
        from vea.highlight import denoise as denoise_grid

        scores = evidence_scores(attribution.attention, attribution.layout, layers)
        if not denoise:
            return scores
        cols, rows = attribution.layout.grid_size
        return denoise_grid(scores.reshape(rows, cols), lam=self.config.lam).flatten()


def answer(
    loaded: LoadedModel,
    image: Image.Image,
    question: str,
    prompt_template: str,
    generation: GenerationConfig | None = None,
) -> tuple[str, int]:
    """Ask ``loaded`` a question about ``image``."""
    inputs = build_inputs(loaded.processor, image, question, prompt_template)
    return generate_answer(
        loaded.model, loaded.processor, inputs.to(loaded.device), generation
    )


def evaluate_sample(
    sample: VQASample,
    method: MethodSpec,
    qa_model: LoadedModel,
    attributor: Attributor | None,
    prompts: dict[str, str],
    generation: GenerationConfig | None = None,
    attribution: Attribution | None = None,
) -> dict:
    """Answer one sample under one method and score the answer.

    Args:
        sample: the VQA sample.
        method: which prompt to use and whether to augment the image.
        qa_model: the model that produces the answer.
        attributor: required when ``method.augment`` is set; may wrap a different
            model than ``qa_model``.
        prompts: prompt templates keyed as in :data:`vea.config.PROMPTS`.
        generation: decoding settings.
        attribution: a cached attribution for this sample, to avoid recomputing
            the forward pass across several methods.

    Returns:
        A result row: identifiers, the generated answer, and the QA metrics.
    """
    if method.augment:
        if attributor is None:
            raise ValueError(f"method {method.name!r} needs an attributor")
        if attribution is None:
            attribution = attributor.attend(sample, prompts["qa"])
        image = attributor.augment(
            attribution,
            denoise=method.denoise,
            smooth=method.smooth,
            use_profiled_layers=method.profile_layers,
        )
    else:
        image = sample.load_image()

    prediction, n_tokens = answer(
        qa_model, image, sample.question, prompts[method.prompt], generation
    )
    scores = score_answer(sample.answers, prediction)

    return {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "model": qa_model.spec.name,
        "method": method.name,
        "image_path": sample.image_path.name,
        "question": sample.question,
        "true_answer": sample.answers[0],
        "model_answer": prediction,
        "n_ans_tokens": n_tokens,
        **scores,
    }


def attribution_row(
    sample: VQASample,
    attribution: Attribution,
    scores: np.ndarray,
    variant: str,
    model_name: str,
) -> dict:
    """Score one set of per-patch evidence scores against the ground truth."""
    return {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "model": model_name,
        "variant": variant,
        "n_bboxes": attribution.labels.n_boxes,
        "n_image_tokens": attribution.layout.n_image_tokens,
        "n_evd_tokens": attribution.labels.n_evidence_patches,
        "evd_token_ratio": attribution.labels.evidence_ratio,
        "image_bbox_oob": attribution.labels.any_out_of_bounds,
        **score_attribution(attribution.labels.patch_mask, scores),
    }


def layer_stats_rows(
    sample: VQASample, attribution: Attribution, model_name: str
) -> list[dict]:
    """Per-layer attention statistics, one row per layer.

    These rows carry the paper's central diagnostic: comparing
    ``image_evd_mean_norm`` against ``image_nonevd_mean_norm`` across depth, and
    splitting by answer correctness, is what shows that deep layers concentrate
    on the evidence even when the final answer is wrong.
    """
    stats = section_attention_stats(
        attribution.attention, attribution.layout, attribution.labels.patch_mask
    )
    return [
        {
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "model": model_name,
            "layer": layer,
            "n_layers": attribution.n_layers,
            "n_image_tokens": attribution.layout.n_image_tokens,
            "n_evd_tokens": attribution.labels.n_evidence_patches,
            "evd_token_ratio": attribution.labels.evidence_ratio,
            **{name: float(values[layer]) for name, values in stats.items()},
        }
        for layer in range(attribution.n_layers)
    ]
