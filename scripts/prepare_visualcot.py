#!/usr/bin/env python3
"""Build a dataset directory from the Visual-CoT benchmark annotations.

Visual-CoT (Shao et al., 2024) supplies the human-annotated evidence boxes this
work relies on. Its metadata files already use the field names this repo expects,
so preparation amounts to filtering, subsampling, and collecting images:

    metadata/{dataset}_cot_{split}.jsonl  ->  data/{dataset}/metadata.json
                                              data/{dataset}/images/*

Usage::

    # 1. Fetch the annotations (small) and the image archives (large) from
    #    https://huggingface.co/datasets/deepcs233/Visual-CoT
    # 2. Point this script at them:
    python scripts/prepare_visualcot.py \\
        --annotations ~/visual-cot/metadata/docvqa_cot_train.jsonl \\
        --images-root ~/visual-cot/images/docvqa \\
        --dataset docvqa --n-samples 100

Note: the bundled ``data/textvqa`` sample was produced by this same procedure, so
that dataset can be used to sanity-check the output format. This script itself has
only been exercised against that layout -- Visual-CoT's per-dataset image
directory names vary, so ``--images-root`` may need adjusting per source.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vea.data import DATA_ROOT

#: Fields carried over verbatim from the Visual-CoT annotations.
KEPT_FIELDS = ("question", "answer", "image", "width", "height", "bboxs", "dataset", "split")


def read_annotations(path: Path) -> list[dict]:
    """Read Visual-CoT annotations from JSON Lines or a JSON array."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def normalize_record(record: dict, dataset: str) -> dict | None:
    """Reduce one annotation to this repo's schema, or drop it.

    Records are dropped when they lack a question, an answer, or an evidence box;
    all three are required by the attribution evaluation.
    """
    boxes = record.get("bboxs") or []
    if not record.get("question") or not record.get("image") or not boxes:
        return None

    answers = record.get("possible_answers") or []
    primary = record.get("answer")
    if primary is None and not answers:
        return None

    out = {key: record[key] for key in KEPT_FIELDS if key in record}
    out["dataset"] = record.get("dataset", dataset)
    # Some source files carry several acceptable answers; keep them all so the
    # QA metrics can credit any valid phrasing.
    if answers:
        out["answers"] = [str(a) for a in answers]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--annotations", type=Path, required=True,
                        help="Visual-CoT metadata file (.jsonl or .json)")
    parser.add_argument("--images-root", type=Path, required=True,
                        help="directory holding the source images")
    parser.add_argument("--dataset", required=True, help="output dataset directory name")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--n-samples", type=int, default=100,
                        help="how many samples to keep (0 keeps all)")
    parser.add_argument("--seed", type=int, default=42, help="subsampling seed")
    parser.add_argument("--symlink", action="store_true",
                        help="symlink images instead of copying them")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be written without writing it")
    args = parser.parse_args(argv)

    records = read_annotations(args.annotations)
    print(f"read {len(records)} annotations from {args.annotations}")

    normalized = [r for r in (normalize_record(r, args.dataset) for r in records) if r]
    print(f"{len(normalized)} have a question, an answer and at least one evidence box")

    # Keep only samples whose image is actually available locally.
    available = [r for r in normalized if (args.images_root / r["image"]).is_file()]
    if not available:
        raise SystemExit(
            f"none of the {len(normalized)} images were found under {args.images_root}. "
            "Visual-CoT stores images per source dataset; check the directory."
        )
    if len(available) < len(normalized):
        print(f"{len(normalized) - len(available)} skipped: image missing under {args.images_root}")

    if args.n_samples and args.n_samples < len(available):
        import random

        random.Random(args.seed).shuffle(available)
        available = available[: args.n_samples]
    print(f"keeping {len(available)} samples")

    dataset_dir = args.data_root / args.dataset
    image_dir = dataset_dir / "images"
    if args.dry_run:
        print(f"[dry run] would write {dataset_dir / 'metadata.json'} "
              f"and {len({r['image'] for r in available})} images into {image_dir}")
        return 0

    image_dir.mkdir(parents=True, exist_ok=True)
    for name in sorted({r["image"] for r in available}):
        source, target = args.images_root / name, image_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        if args.symlink:
            target.symlink_to(source.resolve())
        else:
            shutil.copy2(source, target)

    metadata_path = dataset_dir / "metadata.json"
    metadata_path.write_text(json.dumps(available, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {metadata_path} and {len({r['image'] for r in available})} images to {image_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
