"""Detector benchmark: fit each detector on the train fold, score the test fold, and
report overall ranking quality plus recall broken down by fraud difficulty tier.

The headline number is **recall@prevalence**: flag the top-K test rows by score, where K
equals the number of true frauds in the test fold (i.e. operate the detector at the
dataset's own fraud rate), then measure what fraction of each tier's frauds are caught.
This is threshold-free across detectors and exposes the easy>medium>hard gradient.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fraudlab.evaluation.detectors import build_detectors
from fraudlab.evaluation.features import FeatureBundle, grouped_split


def _recall_at_k_by_tier(
    scores: np.ndarray, y: np.ndarray, tiers: np.ndarray, k: int
) -> dict[str, Any]:
    """Flag the top-k scores; report overall and per-tier recall among flagged rows."""
    n = len(scores)
    k = max(0, min(k, n))
    # Deterministic tie handling: sort by (-score, index).
    order = np.lexsort((np.arange(n), -scores))
    flagged = np.zeros(n, dtype=bool)
    flagged[order[:k]] = True

    result: dict[str, Any] = {}
    fraud_mask = y == 1
    total_fraud = int(fraud_mask.sum())
    result["overall"] = round(
        int((flagged & fraud_mask).sum()) / total_fraud, 4
    ) if total_fraud else 0.0

    for tier in ("easy", "medium", "hard"):
        tmask = (tiers == tier) & fraud_mask
        denom = int(tmask.sum())
        caught = int((flagged & tmask).sum())
        result[tier] = round(caught / denom, 4) if denom else None
    result["_flagged"] = flagged
    return result


def run_benchmark(
    bundle: FeatureBundle, *, test_size: float = 0.3, random_state: int = 42
) -> dict[str, Any]:
    """Train/score every detector and assemble the benchmark report."""
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    train_idx, test_idx = grouped_split(
        bundle, test_size=test_size, random_state=random_state
    )
    X = bundle.X
    y = bundle.y
    tiers = bundle.tiers

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    tiers_test = tiers[test_idx]

    k = int((y_test == 1).sum())  # operate at the test fold's true fraud count

    split_info = {
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_fraud": int(y_train.sum()),
        "test_fraud": int(y_test.sum()),
        "users_disjoint": bool(
            set(np.unique(bundle.groups[train_idx]))
            .isdisjoint(set(np.unique(bundle.groups[test_idx])))
        ),
    }

    detectors_report: dict[str, Any] = {}
    for det in build_detectors(random_state=random_state):
        if getattr(det, "trainable", False):
            det.fit(X_train, y_train)
        scores = det.predict_proba_fraud(X_test)

        roc = float(roc_auc_score(y_test, scores)) if k and k < len(y_test) else float("nan")
        pr = float(average_precision_score(y_test, scores)) if k else float("nan")

        tier_recall = _recall_at_k_by_tier(scores, y_test, tiers_test, k)
        preds = tier_recall.pop("_flagged")
        f1 = float(f1_score(y_test, preds)) if k else 0.0

        detectors_report[det.name] = {
            "roc_auc": round(roc, 4),
            "pr_auc": round(pr, 4),
            "f1_at_prevalence": round(f1, 4),
            "recall_overall": tier_recall["overall"],
            "recall_easy": tier_recall["easy"],
            "recall_medium": tier_recall["medium"],
            "recall_hard": tier_recall["hard"],
        }

    return {
        "protocol": "user-grouped split; recall@prevalence (top-K = test fraud count)",
        "split": split_info,
        "detectors": detectors_report,
    }
