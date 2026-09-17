"""Loading VQA samples with evidence-box annotations.

All datasets share one on-disk layout, so adding a dataset is a matter of
producing its ``metadata.json`` (see ``scripts/prepare_visualcot.py``) rather
than writing new loader code::

    data/
      <dataset>/
        metadata.json
        images/
          <image filename>
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

__all__ = ["VQASample", "load_samples", "available_datasets", "DATA_ROOT"]

#: Default dataset root, resolved relative to the repository.
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class VQASample:
    """One question about one image, with its annotated evidence regions."""

    sample_id: int
    question: str
    answers: list[str]
    """Acceptable answers; the first is the primary reference."""

    image_path: Path
    boxes: list[list[float]] = field(repr=False)
    """Evidence boxes as ``(x0, y0, x1, y1)`` in *original* image pixels."""

    dataset: str = ""
    split: str = ""

    def load_image(self) -> Image.Image:
        """Read the image as RGB.

        Conversion is unconditional: several source images are grayscale or
        palette-mode, and the adapters assume three channels.
        """
        with Image.open(self.image_path) as image:
            return image.convert("RGB")


def available_datasets(root: Path | str = DATA_ROOT) -> list[str]:
    """Names of datasets that are present on disk."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "metadata.json").is_file())


def load_samples(
    dataset: str, root: Path | str = DATA_ROOT, limit: int | None = None
) -> list[VQASample]:
    """Load a dataset's samples.

    Args:
        dataset: dataset directory name, e.g. ``"textvqa"``.
        root: dataset root directory.
        limit: keep only the first ``limit`` samples. Sample ids stay tied to the
            position in the full file, so a truncated run remains comparable to a
            full one.

    Returns:
        The loaded samples, in file order.

    Raises:
        FileNotFoundError: if the metadata file or the image directory is absent.
        ValueError: if a record is missing a required field.
    """
    dataset_dir = Path(root) / dataset
    metadata_path = dataset_dir / "metadata.json"
    image_dir = dataset_dir / "images"

    if not metadata_path.is_file():
        present = available_datasets(root)
        raise FileNotFoundError(
            f"no metadata at {metadata_path}. Datasets present: {present or 'none'}. "
            "See data/README.md for the expected layout."
        )
    if not image_dir.is_dir():
        raise FileNotFoundError(f"no image directory at {image_dir}")

    with metadata_path.open(encoding="utf-8") as f:
        records = json.load(f)

    samples = []
    for sample_id, record in enumerate(records[: limit if limit else None]):
        missing = {"question", "image"} - record.keys()
        if missing:
            raise ValueError(f"{metadata_path}[{sample_id}] is missing {sorted(missing)}")

        # Accept either a single `answer` or a list of `answers`.
        answers = record.get("answers") or ([record["answer"]] if "answer" in record else [])
        if not answers:
            raise ValueError(f"{metadata_path}[{sample_id}] has no answer")

        samples.append(
            VQASample(
                sample_id=sample_id,
                question=record["question"],
                answers=[str(a) for a in answers],
                image_path=image_dir / record["image"],
                boxes=[list(map(float, box)) for box in record.get("bboxs", [])],
                dataset=record.get("dataset", dataset),
                split=record.get("split", ""),
            )
        )
    return samples
