"""Label-leakage detection.

Leakage = a feature that reveals the label in a way a real detector could not exploit at
scoring time. Two kinds are checked:

1. *Construction leakage* — a column derived from the label itself (here ``fraud_type``,
   which is non-null iff the row is fraud). We confirm rather than assume this.
2. *Single-feature leakage* — any individual feature that separates the classes almost
   perfectly (univariate ROC-AUC ~1.0 or mutual information ~1.0). For a realistic fraud
   dataset no single causal feature should be a near-perfect classifier; if one is, the
   generator has accidentally baked the label into a feature.

A clean result is the *desired* outcome: it shows the causal rolling features and per-row
attributes are legitimately usable and that difficulty comes from signal structure, not a
giveaway column.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fraudlab.evaluation.features import (
    LEAKY_BY_CONSTRUCTION,
    build_feature_bundle,
)

# A univariate AUC at/above this (or symmetrically <= 1 - this) is treated as a leak: one
# feature alone almost perfectly orders fraud vs legit.
AUC_LEAK_THRESHOLD = 0.98
# Normalized mutual information at/above this also flags a near-deterministic relationship.
MI_LEAK_THRESHOLD = 0.85


def _univariate_auc(x: np.ndarray, y: np.ndarray) -> float:
    """ROC-AUC of a single feature as a score, folded to [0.5, 1.0] (direction-agnostic)."""
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2 or np.all(x == x[0]):
        return 0.5
    auc = roc_auc_score(y, x)
    return float(max(auc, 1.0 - auc))


def detect_leakage(df: pd.DataFrame) -> dict[str, Any]:
    """Run construction-leakage and single-feature-leakage checks over the dataset."""
    from sklearn.feature_selection import mutual_info_classif

    y = df["is_fraud"].astype(int).to_numpy()

    # --- 1) Construction leakage: fraud_type is non-null exactly for fraud rows. -------
    construction = []
    for col in LEAKY_BY_CONSTRUCTION:
        if col not in df.columns:
            continue
        non_null_is_fraud = df.loc[df[col].notna(), "is_fraud"]
        # Does non-null(col) perfectly coincide with is_fraud == True?
        perfectly_aligned = bool(non_null_is_fraud.all()) and bool(
            (~df.loc[df[col].isna(), "is_fraud"]).all()
        )
        construction.append(
            {
                "column": col,
                "leaks_label": perfectly_aligned,
                "reason": "non-null exactly on fraud rows (label proxy)"
                if perfectly_aligned
                else "non-null/null does not align with label",
                "excluded_from_features": True,
            }
        )

    # --- 2) Single-feature leakage over the *usable* feature matrix. -------------------
    bundle = build_feature_bundle(df)
    X = bundle.X
    feature_cols = list(X.columns)

    # Mutual information (normalized by label entropy) per feature.
    mi = mutual_info_classif(
        X.to_numpy(), y, discrete_features=False, random_state=0
    )
    p = y.mean()
    label_entropy = -(p * np.log(p + 1e-12) + (1 - p) * np.log(1 - p + 1e-12))
    mi_norm = mi / (label_entropy + 1e-12)

    per_feature = []
    flagged = []
    for col, mi_val in zip(feature_cols, mi_norm):
        auc = _univariate_auc(X[col].to_numpy().astype(float), y)
        is_leak = auc >= AUC_LEAK_THRESHOLD or mi_val >= MI_LEAK_THRESHOLD
        rec = {
            "feature": col,
            "univariate_auc": round(auc, 4),
            "norm_mutual_info": round(float(mi_val), 4),
            "leak": bool(is_leak),
        }
        per_feature.append(rec)
        if is_leak:
            flagged.append(rec)

    per_feature.sort(key=lambda r: r["univariate_auc"], reverse=True)

    return {
        "construction_leakage": construction,
        "single_feature_leakage": flagged,
        "n_features_checked": len(feature_cols),
        "auc_threshold": AUC_LEAK_THRESHOLD,
        "mi_threshold": MI_LEAK_THRESHOLD,
        "top_features_by_auc": per_feature[:8],
        "clean": len(flagged) == 0,
    }
