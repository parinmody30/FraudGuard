"""Benchmark metrics: ranking quality, thresholded scores, tiered recall, and top-K recall.

All detectors expose a continuous fraud *score* (higher = riskier) plus a binary *label*
at their natural operating point. ROC-AUC / PR-AUC / top-K / tiered recall are computed
from the score (threshold-free, cross-model comparable); precision / recall / F1 are
computed from the binary label.

Tiered recall (easy/medium/hard) is measured at the *prevalence operating point*: flag the
top-K highest-scoring rows where K equals the number of true frauds in the test fold. This
mirrors how Phase 1.1 isolates per-tier detectability and keeps the comparison fair across
models with very different score scales.
"""

from __future__ import annotations

from typing import Any

import numpy as np

TIERS: tuple[str, ...] = ("easy", "medium", "hard")
TOPK_FRACTIONS: tuple[float, ...] = (0.01, 0.02, 0.05)


def _flag_top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Return a boolean mask flagging the ``k`` highest scores (deterministic tie-break)."""
    n = len(scores)
    k = max(0, min(k, n))
    flagged = np.zeros(n, dtype=bool)
    if k:
        order = np.lexsort((np.arange(n), -scores))  # high score first, stable on ties
        flagged[order[:k]] = True
    return flagged


def recall_at_fraction(scores: np.ndarray, y: np.ndarray, fraction: float) -> float:
    """Recall when reviewing the top ``fraction`` of all rows by score."""
    n = len(scores)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    k = int(np.ceil(fraction * n))
    flagged = _flag_top_k(scores, k)
    return float((flagged & (y == 1)).sum() / n_pos)


def tiered_recall(
    scores: np.ndarray, y: np.ndarray, tiers: np.ndarray, k: int
) -> dict[str, float | None]:
    """Per-tier recall among the top-``k`` flagged rows."""
    flagged = _flag_top_k(scores, k)
    out: dict[str, float | None] = {}
    for tier in TIERS:
        mask = (tiers == tier) & (y == 1)
        denom = int(mask.sum())
        out[tier] = round(float((flagged & mask).sum() / denom), 4) if denom else None
    return out


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    pred_labels: np.ndarray,
    tiers: np.ndarray,
) -> dict[str, Any]:
    """Compute the full benchmark metric set for one model on one test fold."""
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    pred_labels = np.asarray(pred_labels).astype(int)
    n_pos = int(y_true.sum())
    both_classes = len(np.unique(y_true)) == 2

    metrics: dict[str, Any] = {
        "n": int(len(y_true)),
        "n_positive": n_pos,
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4) if both_classes else float("nan"),
        "pr_auc": round(float(average_precision_score(y_true, scores)), 4) if both_classes else float("nan"),
        "precision": round(float(precision_score(y_true, pred_labels, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred_labels, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred_labels, zero_division=0)), 4),
    }

    tr = tiered_recall(scores, y_true, tiers, n_pos)
    metrics["recall_easy"] = tr["easy"]
    metrics["recall_medium"] = tr["medium"]
    metrics["recall_hard"] = tr["hard"]

    for frac in TOPK_FRACTIONS:
        label = f"recall_at_{int(frac * 100)}pct"
        metrics[label] = round(recall_at_fraction(scores, y_true, frac), 4)

    return metrics
