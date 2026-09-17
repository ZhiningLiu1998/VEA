#!/usr/bin/env python3
"""Attribution accuracy: how well does attention localise the annotated evidence?

Compares Vea's profiled-layer scores against fixed depth ranges, measuring how
well each ranks evidence patches above non-evidence ones (AUROC, NDCG).

    python scripts/run_attribution.py --model qwen2.5-vl-7b --datasets textvqa \
        --profile results/profiles/qwen2.5-vl-7b.json

Reproduces the paper's attribution table for the layer-span rows and the Vea row.
The VAR and AGLA rows are not implemented here; see the README.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vea.attention import evidence_scores, layer_span_indices
from vea.cli import (
    RESULTS_ROOT,
    add_data_args,
    add_model_args,
    add_vea_args,
    progress,
    resolve_dtype,
    vea_config_from_args,
    write_rows,
)
from vea.config import PROMPTS
from vea.data import load_samples
from vea.models import load_model
from vea.pipeline import Attributor, attribution_row
from vea.profiling import load_profile

#: Fixed depth ranges to compare against the profiled selection. The contrast
#: between the two halves is what shows that visual grounding lives deep.
LAYER_SPANS: dict[str, tuple[float, float]] = {
    "L_0-100": (0.0, 1.0),
    "L_0-50": (0.0, 0.5),
    "L_50-100": (0.5, 1.0),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(parser)
    add_data_args(parser)
    add_vea_args(parser)
    parser.add_argument(
        "--profile", type=Path, default=None,
        help="layer profile JSON. Without it, only the fixed spans are evaluated.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = vea_config_from_args(args)

    loaded = load_model(
        args.model, device=args.device, dtype=resolve_dtype(args.dtype),
        eager_attention=True, vea_config=config,
    )
    profile = load_profile(args.profile) if args.profile else None
    attributor = Attributor(loaded, layers=profile.layers if profile else None, config=config)
    if profile is None:
        print("No --profile given: evaluating the fixed layer spans only.")

    output_dir = Path(args.output_dir or RESULTS_ROOT / "attribution")
    for dataset in args.datasets:
        samples = load_samples(dataset, root=args.data_root, limit=args.limit)
        print(f"\n{dataset}: {len(samples)} samples")

        rows = []
        for sample in progress(samples, description=dataset, total=len(samples)):
            if not sample.boxes:
                continue
            attribution = attributor.attend(sample, PROMPTS["qa"])

            variants = {
                name: evidence_scores(
                    attribution.attention,
                    attribution.layout,
                    layer_span_indices(attribution.n_layers, span),
                )
                for name, span in LAYER_SPANS.items()
            }
            if profile is not None:
                # Vea's own score includes denoising, since that is what the
                # method actually feeds downstream.
                variants["vea"] = attributor.patch_scores(
                    attribution, profile.layers, denoise=True
                )

            rows.extend(
                attribution_row(sample, attribution, scores, name, loaded.spec.name)
                for name, scores in variants.items()
            )

        path = write_rows(rows, output_dir / f"{loaded.spec.name}__{dataset}.csv")
        print(f"wrote {path}")

        import pandas as pd

        order = [*LAYER_SPANS, *(["vea"] if profile is not None else [])]
        summary = (
            pd.DataFrame(rows)
            .groupby("variant")[["auroc", "ndcg"]]
            .mean()
            .mul(100)
            .round(2)
            .reindex(order)
        )
        print(summary.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
