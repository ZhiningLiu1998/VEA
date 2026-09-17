#!/usr/bin/env python3
"""The paper's central diagnostic: "seeing but not believing".

For every sample this records, per layer, how much attention goes to the image,
to the question, to annotated evidence patches, and to non-evidence patches --
and whether the model's answer was correct.

Two findings fall out of the resulting table:

* Attention shifts from text to image with depth, and deep layers concentrate
  sharply on the evidence patches (the layer-dynamics figures).
* That concentration is nearly as strong when the answer is *wrong* as when it is
  right. The models locate the evidence and still fail to use it -- perception is
  not the bottleneck (the correctness-split figure).

    python scripts/run_analysis.py --model qwen2.5-vl-7b --datasets textvqa

Writes one row per (sample, layer) plus a printed summary contrasting correct and
incorrect answers over the deeper half of the stack.
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
from vea.config import PROMPTS, GenerationConfig
from vea.data import load_samples
from vea.metrics import score_answer
from vea.models import load_model
from vea.pipeline import Attributor, answer, layer_stats_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_model_args(parser)
    add_data_args(parser)
    add_vea_args(parser)
    parser.add_argument(
        "--skip-answers", action="store_true",
        help="record attention only, without generating answers. Halves the "
             "compute but loses the correctness split.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = vea_config_from_args(args)
    generation = GenerationConfig()

    loaded = load_model(
        args.model, device=args.device, dtype=resolve_dtype(args.dtype),
        eager_attention=True, vea_config=config,
    )
    attributor = Attributor(loaded, layers=None, config=config)

    output_dir = Path(args.output_dir or RESULTS_ROOT / "analysis")
    for dataset in args.datasets:
        samples = load_samples(dataset, root=args.data_root, limit=args.limit)
        print(f"\n{dataset}: {len(samples)} samples")

        rows = []
        for sample in progress(samples, description=dataset, total=len(samples)):
            if not sample.boxes:
                continue
            attribution = attributor.attend(sample, PROMPTS["qa"])

            correctness: dict[str, object] = {}
            if not args.skip_answers:
                prediction, _ = answer(
                    loaded, sample.load_image(), sample.question, PROMPTS["qa"], generation
                )
                scores = score_answer(sample.answers, prediction)
                correctness = {
                    "model_answer": prediction,
                    "true_answer": sample.answers[0],
                    "em": scores["em"],
                    "f1": scores["f1"],
                }

            rows.extend(
                {**row, **correctness}
                for row in layer_stats_rows(sample, attribution, loaded.spec.name)
            )

        path = write_rows(rows, output_dir / f"{loaded.spec.name}__{dataset}.csv")
        print(f"wrote {path}")
        _print_summary(rows, skip_answers=args.skip_answers)

    return 0


def _print_summary(rows: list[dict], skip_answers: bool) -> None:
    """Contrast evidence and non-evidence attention over the deeper half."""
    import pandas as pd

    if not rows:
        print("no samples with evidence boxes; nothing to summarize")
        return

    frame = pd.DataFrame(rows)
    deep = frame[frame["layer"] >= frame["n_layers"] / 2]
    columns = [
        "image_mean_norm",
        "quest_mean_norm",
        "image_evd_mean_norm",
        "image_nonevd_mean_norm",
    ]

    print("\nRelative attention per token (RAPT), deeper half of the stack:")
    if skip_answers:
        print(deep[columns].mean().round(3).to_string())
        return

    grouped = deep.groupby(deep["em"] > 0)[columns].mean().round(3)
    grouped.index = grouped.index.map({True: "correct", False: "incorrect"})
    print(grouped.to_string())

    ratio = deep["image_evd_mean_norm"] / deep["image_nonevd_mean_norm"]
    by_correct = ratio.groupby(deep["em"] > 0).mean()
    print("\nEvidence / non-evidence attention ratio:")
    for correct, value in by_correct.items():
        print(f"  {'correct' if correct else 'incorrect':<10} {value:.2f}x")
    print(
        "\nA ratio well above 1.0 for *incorrect* answers is the paper's point: "
        "the evidence was found but not used."
    )


if __name__ == "__main__":
    raise SystemExit(main())
