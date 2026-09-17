#!/usr/bin/env python3
"""Main QA experiment: Base vs. Inst vs. Vea, plus Vea's ablations.

Answer accuracy (Exact Match, token F1) for each method on each dataset.

    # Main comparison
    python scripts/run_qa.py --model qwen2.5-vl-7b --datasets textvqa \
        --methods base inst vea --profile results/profiles/qwen2.5-vl-7b.json

    # Ablations
    python scripts/run_qa.py --model qwen2.5-vl-7b --datasets textvqa \
        --methods vea vea-no-denoise vea-no-smooth vea-no-profiling \
        --profile results/profiles/qwen2.5-vl-7b.json

    # Delegate attribution: Qwen locates the evidence, InternVL answers
    python scripts/run_qa.py --model internvl3.5-8b --attribution-model qwen2.5-vl-7b \
        --profile results/profiles/qwen2.5-vl-7b.json --methods base vea

Reproduces the paper's main QA table (Base/Inst/Vea columns) and its ablation
table. The CGR, VAR and AGLA baselines are not implemented here; see the README.
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
    write_rows,
)
from vea.config import METHODS, PROMPTS, GenerationConfig, resolve_method
from vea.data import load_samples
from vea.models import load_model
from vea.pipeline import Attributor, evaluate_sample
from vea.profiling import load_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(parser)
    add_data_args(parser)
    add_vea_args(parser)
    parser.add_argument(
        "--methods", nargs="+", default=["base", "inst", "vea"], choices=sorted(METHODS),
        help="methods to evaluate",
    )
    parser.add_argument(
        "--attribution-model", default=None,
        help="model used to locate evidence and build the augmented image. "
             "Defaults to --model. Use a delegate when --model cannot expose "
             "attention within the available memory budget.",
    )
    parser.add_argument(
        "--profile", type=Path, default=None,
        help="layer profile JSON from profile_layers.py. Without it, Vea averages "
             "over all layers, which is the w/o-Profiling setting.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=GenerationConfig().max_new_tokens)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = vea_config_from_args(args)
    dtype = resolve_dtype(args.dtype)
    generation = GenerationConfig(max_new_tokens=args.max_new_tokens)

    methods = [resolve_method(name) for name in args.methods]
    needs_attribution = any(method.augment for method in methods)

    # The answering model only needs attention when it also attributes.
    attribution_model_name = args.attribution_model or args.model
    delegated = attribution_model_name != args.model

    qa_model = load_model(
        args.model, device=args.device, dtype=dtype,
        eager_attention=needs_attribution and not delegated, vea_config=config,
    )

    attributor = None
    if needs_attribution:
        attribution_model = (
            load_model(attribution_model_name, device=args.device, dtype=dtype,
                       eager_attention=True, vea_config=config)
            if delegated
            else qa_model
        )
        layers = load_profile(args.profile).layers if args.profile else None
        if layers is None:
            print("No --profile given: Vea will average over all layers.")
        attributor = Attributor(attribution_model, layers=layers, config=config)
        print(
            f"Attribution: {attribution_model.spec.name} "
            f"(layers={attributor.profiled_layers if args.profile else 'all'})"
            + (" [delegate]" if delegated else "")
        )

    output_dir = Path(args.output_dir or RESULTS_ROOT / "qa")
    for dataset in args.datasets:
        samples = load_samples(dataset, root=args.data_root, limit=args.limit)
        print(f"\n{dataset}: {len(samples)} samples x {len(methods)} methods")

        rows = []
        for sample in progress(samples, description=dataset, total=len(samples)):
            # One attribution pass per sample, reused by every augmenting method.
            attribution = (
                attributor.attend(sample, PROMPTS["qa"]) if needs_attribution else None
            )
            for method in methods:
                rows.append(
                    evaluate_sample(
                        sample, method, qa_model, attributor, PROMPTS,
                        generation=generation, attribution=attribution,
                    )
                )

        path = write_rows(rows, output_dir / f"{qa_model.spec.name}__{dataset}.csv")
        print(f"wrote {path}")

        import pandas as pd

        summary = (
            pd.DataFrame(rows)
            .groupby("method")[["em", "f1"]]
            .mean()
            .mul(100)
            .round(2)
            .reindex([m.name for m in methods])
        )
        print(summary.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
