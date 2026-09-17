#!/usr/bin/env python3
"""Aggregate per-run CSVs into the paper's tables.

    python scripts/summarize.py qa            # accuracy by model x dataset x method
    python scripts/summarize.py attribution   # AUROC / NDCG by layer selection
    python scripts/summarize.py analysis      # attention split by answer correctness

Reads whatever the run scripts left under ``results/`` and prints a table; pass
``--csv PATH`` to also save it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from vea.cli import RESULTS_ROOT

#: Which columns to aggregate and how to group them, per experiment.
VIEWS = {
    "qa": {"metrics": ["em", "f1"], "index": ["model", "dataset"], "columns": "method"},
    "attribution": {
        "metrics": ["auroc", "ndcg"],
        "index": ["model", "dataset"],
        "columns": "variant",
    },
}


def load_results(kind: str, root: Path) -> pd.DataFrame:
    """Concatenate every CSV written for one experiment kind."""
    directory = root / kind
    paths = sorted(directory.glob("*.csv"))
    if not paths:
        raise SystemExit(f"no CSV files under {directory}; run the corresponding script first")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def summarize_table(frame: pd.DataFrame, view: dict) -> pd.DataFrame:
    """Pivot per-sample rows into a model x dataset by method table, in percent."""
    table = frame.pivot_table(
        index=view["index"], columns=view["columns"], values=view["metrics"], aggfunc="mean"
    )
    return (table * 100).round(2)


def summarize_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    """Evidence vs. non-evidence attention over the deeper half, split by correctness."""
    deep = frame[frame["layer"] >= frame["n_layers"] / 2].copy()
    columns = [
        c
        for c in (
            "image_mean_norm",
            "quest_mean_norm",
            "image_evd_mean_norm",
            "image_nonevd_mean_norm",
        )
        if c in deep.columns
    ]
    index = ["model", "dataset"]
    if "em" in deep.columns:
        deep["answer"] = (deep["em"] > 0).map({True: "correct", False: "incorrect"})
        index = index + ["answer"]
    table = deep.groupby(index)[columns].mean().round(3)
    table["evd_over_nonevd"] = (
        table["image_evd_mean_norm"] / table["image_nonevd_mean_norm"]
    ).round(2)
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("kind", choices=["qa", "attribution", "analysis"])
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--csv", type=Path, default=None, help="also write the table here")
    args = parser.parse_args(argv)

    frame = load_results(args.kind, args.results_root)
    table = (
        summarize_analysis(frame)
        if args.kind == "analysis"
        else summarize_table(frame, VIEWS[args.kind])
    )

    n_samples = frame["sample_id"].nunique()
    print(f"{args.kind}: {len(frame)} rows over {n_samples} unique samples\n")
    print(table.to_string())

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.csv)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
