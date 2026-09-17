"""Visual-grounding layer profiling (Vea Step A).

Layers differ sharply in how well their attention localises visual evidence, and
which layers are best is a property of the model, not of the input.  Profiling
therefore runs once per model on a small diagnostic set and yields a reusable set
of layer indices.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from vea.config import PROMPTS, ProfileResult, VeaConfig
from vea.data import VQASample
from vea.metrics import score_attribution
from vea.pipeline import Attributor

__all__ = ["profile_layers", "save_profile", "load_profile"]


def profile_layers(
    attributor: Attributor,
    samples: list[VQASample],
    config: VeaConfig | None = None,
    progress=None,
) -> ProfileResult:
    """Rank layers by how well their attention localises annotated evidence.

    Each layer's per-patch attention is scored against the binary evidence
    labels with AUROC, averaged over the diagnostic samples; the top
    ``layer_top_fraction`` of layers is retained.  AUROC is used rather than a
    threshold-based measure because only the *ranking* of patches matters -- the
    absolute attention scale varies by orders of magnitude across layers.

    Args:
        attributor: attributor wrapping the model to profile.
        samples: diagnostic samples. The paper uses 100 TextVQA examples and
            notes that roughly this many suffice for a stable selection.
        config: supplies ``layer_top_fraction``.
        progress: optional callable invoked with each processed sample index.

    Returns:
        The :class:`~vea.config.ProfileResult`.

    Raises:
        ValueError: if no sample yielded a usable score, which happens when every
            sample lacks evidence boxes or has all patches labelled alike.
    """
    config = config or attributor.config
    per_sample: list[np.ndarray] = []

    for index, sample in enumerate(samples):
        if not sample.boxes:
            continue
        attribution = attributor.attend(sample, PROMPTS["qa"])
        labels = attribution.labels.patch_mask
        if labels.min() == labels.max():
            # Degenerate: AUROC is undefined when all patches share a label.
            continue

        start, end = attribution.layout.image_span
        image_attention = attribution.attention[:, start:end]
        per_sample.append(
            np.array(
                [
                    score_attribution(labels, image_attention[layer])["auroc"]
                    for layer in range(image_attention.shape[0])
                ]
            )
        )
        if progress is not None:
            progress(index)

    if not per_sample:
        raise ValueError(
            "no usable diagnostic samples: every sample either lacked evidence "
            "boxes or had all patches labelled identically"
        )

    layer_auroc = np.nanmean(np.stack(per_sample), axis=0)
    n_layers = int(layer_auroc.shape[0])
    n_keep = max(1, math.ceil(config.layer_top_fraction * n_layers))
    # argsort is ascending, so the last n_keep entries are the highest scoring.
    selected = sorted(int(i) for i in np.argsort(layer_auroc)[-n_keep:])

    return ProfileResult(
        model=attributor.loaded.spec.name,
        n_layers=n_layers,
        layers=selected,
        layer_auroc=[float(v) for v in layer_auroc],
        n_samples=len(per_sample),
        dataset=samples[0].dataset if samples else "",
    )


def save_profile(profile: ProfileResult, path: Path | str) -> Path:
    """Write a profile to JSON, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": profile.model,
        "dataset": profile.dataset,
        "n_samples": profile.n_samples,
        "n_layers": profile.n_layers,
        "layers": profile.layers,
        "mean_auroc_all_layers": profile.mean_auroc_all,
        "mean_auroc_selected_layers": profile.mean_auroc_selected,
        "layer_auroc": profile.layer_auroc,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_profile(path: Path | str) -> ProfileResult:
    """Read a profile written by :func:`save_profile`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ProfileResult(
        model=payload["model"],
        n_layers=payload["n_layers"],
        layers=list(payload["layers"]),
        layer_auroc=list(payload["layer_auroc"]),
        n_samples=payload.get("n_samples", 0),
        dataset=payload.get("dataset", ""),
    )
