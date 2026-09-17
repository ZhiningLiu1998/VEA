"""Shared command-line plumbing for the experiment scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from vea.config import VeaConfig
from vea.data import DATA_ROOT

__all__ = [
    "RESULTS_ROOT",
    "add_data_args",
    "add_model_args",
    "add_vea_args",
    "vea_config_from_args",
    "resolve_dtype",
    "write_rows",
    "progress",
]

#: Default output root for every script.
RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def add_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Dataset selection and output location."""
    group = parser.add_argument_group("data")
    group.add_argument(
        "--datasets", nargs="+", default=["textvqa"], help="dataset directory names"
    )
    group.add_argument("--data-root", type=Path, default=DATA_ROOT, help="dataset root")
    group.add_argument(
        "--limit", type=int, default=None, help="evaluate only the first N samples"
    )
    group.add_argument(
        "--output-dir", type=Path, default=None, help="where to write result files"
    )
    return parser


def add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Model selection and placement."""
    group = parser.add_argument_group("model")
    group.add_argument(
        "--model", required=True, help="short name from vea.config.MODELS, or an HF id"
    )
    group.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    group.add_argument(
        "--dtype", default="bfloat16", choices=sorted(_DTYPES), help="model weight dtype"
    )
    return parser


def add_vea_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Vea hyperparameters. Defaults match the paper."""
    defaults = VeaConfig()
    group = parser.add_argument_group("vea hyperparameters")
    group.add_argument("--alpha", type=float, default=defaults.alpha,
                       help="brightness floor for non-evidence pixels")
    group.add_argument("--sigma", type=float, default=defaults.sigma,
                       help="Gaussian bandwidth, relative to the shorter image side")
    group.add_argument("--lam", type=float, default=defaults.lam,
                       help="denoising spike threshold")
    group.add_argument("--layer-top-fraction", type=float, default=defaults.layer_top_fraction,
                       help="fraction of layers kept when profiling")
    group.add_argument("--overlap-rule", default=defaults.overlap_rule,
                       choices=["any", "half", "all"],
                       help="how much of a patch must be covered to count as evidence")
    group.add_argument("--max-pixels", type=int, default=defaults.max_pixels,
                       help="cap on processor input resolution")
    return parser


def vea_config_from_args(args: argparse.Namespace) -> VeaConfig:
    """Build a :class:`~vea.config.VeaConfig` from parsed arguments."""
    return VeaConfig(
        lam=args.lam,
        sigma=args.sigma,
        alpha=args.alpha,
        layer_top_fraction=args.layer_top_fraction,
        overlap_rule=args.overlap_rule,
        max_pixels=args.max_pixels,
    )


def resolve_dtype(name: str) -> torch.dtype:
    """Map a dtype name to a torch dtype."""
    return _DTYPES[name]


def write_rows(rows: list[dict], path: Path) -> Path:
    """Write result rows to CSV, creating parent directories as needed."""
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def progress(iterable, description: str = "", total: int | None = None):
    """Wrap an iterable in a progress bar, degrading gracefully without tqdm."""
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=description, total=total)
