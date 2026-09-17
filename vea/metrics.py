"""Answer-correctness and evidence-attribution metrics."""

from __future__ import annotations

import numpy as np
from qa_metrics.em import em_match
from qa_metrics.f1 import f1_match, f1_score_with_precision_recall
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score

__all__ = ["QA_METRICS", "ATTRIBUTION_METRICS", "score_answer", "score_attribution"]

#: Answer-correctness keys returned by :func:`score_answer`.
QA_METRICS = ("em", "f1m", "f1", "pr", "re")

#: Attribution keys returned by :func:`score_attribution`.
ATTRIBUTION_METRICS = ("auroc", "auprc", "ndcg")


def _score_single(reference: str, prediction: str) -> dict[str, float]:
    """Score a prediction against one reference answer.

    Note on ``pr``/``re``: ``qa_metrics`` labels these the opposite way round from
    the usual convention -- its "precision" is the fraction of *reference* tokens
    recovered and its "recall" the fraction of *predicted* tokens that are
    correct. The upstream values are passed through unchanged so that ``f1``
    matches published numbers; only the two component names are counter-intuitive.
    """
    scores = f1_score_with_precision_recall(reference, prediction)
    return {
        "em": float(em_match(reference, prediction)),
        "f1m": float(f1_match(reference, prediction)),
        "f1": scores["f1"],
        "pr": scores["precision"],
        "re": scores["recall"],
    }


def score_answer(
    references: list[str], prediction: str, select_by: str = "f1"
) -> dict[str, float]:
    """Score a prediction against a list of acceptable answers.

    Token-level metrics are reported for the single best-matching reference,
    following standard VQA practice. Exact match is evaluated against the whole
    reference set at once so that any acceptable phrasing counts as a hit.

    Args:
        references: acceptable ground-truth answers; must be non-empty.
        prediction: the model's generated answer.
        select_by: which metric picks the best-matching reference.

    Returns:
        A dict with the keys in :data:`QA_METRICS`.
    """
    if not references:
        raise ValueError("references must not be empty")
    if select_by not in QA_METRICS:
        raise ValueError(f"select_by must be one of {QA_METRICS}, got {select_by!r}")

    best = max(
        (_score_single(ref, prediction) for ref in references),
        key=lambda scores: scores[select_by],
    )
    best["em"] = float(em_match(references, prediction))
    return best


def score_attribution(
    labels: np.ndarray, scores: np.ndarray, k: int | None = None
) -> dict[str, float]:
    """Rank evidence patches by attention score and measure the ranking quality.

    Args:
        labels: ``(n_patches,)`` binary array, ``True`` for evidence patches.
        scores: ``(n_patches,)`` predicted evidence score per patch.
        k: truncation for NDCG@k. ``None`` evaluates the full ranking.

    Returns:
        A dict with the keys in :data:`ATTRIBUTION_METRICS`, or ``nan`` values
        when the sample is degenerate (all patches share the same label, which
        leaves AUROC and NDCG undefined).
    """
    labels = np.asarray(labels).astype(int).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    if labels.shape != scores.shape:
        raise ValueError(f"shape mismatch: labels {labels.shape} vs scores {scores.shape}")

    if labels.min() == labels.max():
        return {name: float("nan") for name in ATTRIBUTION_METRICS}

    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "ndcg": float(ndcg_score(labels[None, :], scores[None, :], k=k)),
    }
