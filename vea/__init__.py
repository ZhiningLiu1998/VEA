"""Vea: Visual Evidence Augmentation for vision-language models.

Reference implementation for *Seeing but Not Believing: Probing the Disconnect
Between Visual Attention and Answer Correctness in VLMs* (ICLR 2026).

Vea is a training-free, inference-time method. One forward pass reveals where a
VLM's deep layers already attend, that signal is turned into an evidence map, and
the image is re-shown with the evidence highlighted -- no weights are touched.

Typical use::

    from vea import Attributor, VeaConfig, load_model, load_samples

    model = load_model("qwen2.5-vl-7b")
    attributor = Attributor(model, layers=[18, 22, 24], config=VeaConfig())
    sample = load_samples("textvqa")[0]

    attribution = attributor.attend(sample, PROMPTS["qa"])
    augmented = attributor.augment(attribution)
"""

from vea.config import (
    DATASETS,
    METHODS,
    MODELS,
    PROMPTS,
    GenerationConfig,
    MethodSpec,
    ModelSpec,
    ProfileResult,
    VeaConfig,
    resolve_method,
    resolve_model,
)
from vea.data import VQASample, available_datasets, load_samples
from vea.models import load_model
from vea.pipeline import Attribution, Attributor, answer, evaluate_sample
from vea.profiling import load_profile, profile_layers, save_profile

__version__ = "1.0.0"

__all__ = [
    "DATASETS",
    "METHODS",
    "MODELS",
    "PROMPTS",
    "Attribution",
    "Attributor",
    "GenerationConfig",
    "MethodSpec",
    "ModelSpec",
    "ProfileResult",
    "VQASample",
    "VeaConfig",
    "answer",
    "available_datasets",
    "evaluate_sample",
    "load_model",
    "load_profile",
    "load_samples",
    "profile_layers",
    "resolve_method",
    "resolve_model",
    "save_profile",
]
