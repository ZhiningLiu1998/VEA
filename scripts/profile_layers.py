#!/usr/bin/env python3
"""Vea Step A: find a model's visual-grounding layers.

Scores every layer by how well its attention ranks annotated evidence patches
(AUROC) on a small diagnostic set, and keeps the top-scoring fraction. Run once
per model; the resulting JSON is consumed by every other script.

    python scripts/profile_layers.py --model qwen2.5-vl-7b --limit 100

Reproduces the paper's layer-profiling table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vea.cli import (
    RESULTS_ROOT,
    add_data_args,
    add_model_args,
    add_vea_args,
    progress,
    resolve_dtype,
    vea_config_from_args,
)
from vea.data import load_samples
from vea.models import load_model
from vea.pipeline import Attributor
from vea.profiling import profile_layers, save_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(parser)
    add_data_args(parser)
    add_vea_args(parser)
    parser.set_defaults(datasets=["textvqa"], limit=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.datasets) != 1:
        raise SystemExit("profiling uses a single diagnostic dataset; pass one --datasets value")

    config = vea_config_from_args(args)
    dataset = args.datasets[0]
    samples = load_samples(dataset, root=args.data_root, limit=args.limit)
    print(f"Profiling {args.model} on {len(samples)} {dataset} samples")

    loaded = load_model(
        args.model, device=args.device, dtype=resolve_dtype(args.dtype),
        eager_attention=True, vea_config=config,
    )
    attributor = Attributor(loaded, layers=None, config=config)

    bar = progress(samples, description="profiling", total=len(samples))
    profile = profile_layers(attributor, list(bar), config=config)

    output_dir = args.output_dir or RESULTS_ROOT / "profiles"
    path = save_profile(profile, Path(output_dir) / f"{loaded.spec.name}.json")

    print(
        f"\n{loaded.spec.name}: {profile.n_layers} layers, "
        f"mean AUROC {profile.mean_auroc_all * 100:.2f} over all layers\n"
        f"selected {len(profile.layers)} layers {profile.layers}, "
        f"mean AUROC {profile.mean_auroc_selected * 100:.2f}\n"
        f"wrote {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
