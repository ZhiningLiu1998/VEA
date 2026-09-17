# Datasets

Every dataset lives in its own directory with the same two-part layout:

```
data/
  <dataset>/
    metadata.json      # list of records, one per question
    images/            # image files referenced by those records
```

## Record schema

`metadata.json` is a JSON array. Each record describes one question about one
image, together with the regions that contain the answer:

| Field | Type | Required | Meaning |
|---|---|---|---|
| `question` | string | yes | The question text, passed to the model verbatim. |
| `answer` | string | yes* | The reference answer. |
| `answers` | list of string | no | Several acceptable answers. Takes precedence over `answer`; any of them counts as a hit. |
| `image` | string | yes | Filename inside `images/`. |
| `width`, `height` | int | no | Original image size, in pixels. |
| `bboxs` | list of `[x0, y0, x1, y1]` | yes for attribution | Evidence regions, in **absolute pixels of the original image**, `xyxy` order. |
| `dataset` | string | no | Source dataset name. |
| `split` | string | no | Source split. |

\* either `answer` or a non-empty `answers` is required.

Records without `bboxs` are still usable for question answering, but they are
skipped by the attribution and layer-profiling scripts, which need ground-truth
evidence to score against.

Sample ids are positional: the *n*-th record in the file is `sample_id = n`. This
holds even under `--limit`, so a truncated run stays comparable to a full one.

## What is included here

`data/textvqa/` holds a 50-question sample of TextVQA (37 images), enough to run
every script end to end and to check output formats. It is a demonstration
subset, **not** the evaluation set behind the numbers in the paper.

Images originate from [TextVQA](https://textvqa.org/), whose photographs come
from [Open Images](https://storage.googleapis.com/openimages/web/index.html)
(CC BY 4.0). Annotations are redistributed from Visual-CoT.

## Adding the other datasets

The paper evaluates on TextVQA, DocVQA, SROIE and InfoVQA, taking evidence boxes
for all four from the
[Visual-CoT benchmark](https://huggingface.co/datasets/deepcs233/Visual-CoT)
(Shao et al., 2024). Visual-CoT's own metadata already uses the field names above,
so preparation is mostly filtering and collecting images:

```bash
# 1. Download the annotations (small) and image archives (large) from Visual-CoT.
# 2. Convert one source dataset at a time:
python scripts/prepare_visualcot.py \
    --annotations path/to/metadata/docvqa_cot_train.jsonl \
    --images-root path/to/images/docvqa \
    --dataset docvqa \
    --n-samples 100
```

Add `--dry-run` first to see what would be written. Visual-CoT's image directory
names vary per source dataset, so `--images-root` usually needs adjusting; the
script fails loudly if none of the referenced images are found there.

Because we cannot redistribute the full benchmark, the exact evaluation subsets
used in the paper are not included. Sampling with a different seed or size will
shift absolute numbers, though the ranking of methods is stable.
