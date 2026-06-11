"""Descriptive dataset metrics: shape, class balance, tier mix, cardinality, distributions.

These metrics characterize the dataset independently of any model. They feed the human
report and several downstream checks (e.g. cardinality informs CTGAN readiness).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from fraudlab.evaluation.features import (
    CATEGORICAL_FEATURES,
    ID_COLUMNS,
    NUMERIC_FEATURES,
    fraud_type_to_tier,
    tier_of_row,
)


def _series_stats(s: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {}
    return {
        "min": float(s.min()),
        "p50": float(s.median()),
        "mean": float(s.mean()),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
        "std": float(s.std(ddof=0)),
    }


def dataset_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute descriptive metrics for a transaction dataset."""
    n = len(df)
    n_fraud = int(df["is_fraud"].sum())
    n_legit = n - n_fraud

    mapping = fraud_type_to_tier()
    tiers = [tier_of_row(ft, mapping) for ft in df.get("fraud_type", pd.Series([None] * n))]
    tier_counts = Counter(t for t in tiers if t != "legit")
    fraud_type_counts = Counter(
        str(ft) for ft in df.loc[df["is_fraud"], "fraud_type"].tolist()
    )

    # Per-tier share of *fraud* (not of the whole dataset).
    tier_share = {
        tier: round(tier_counts.get(tier, 0) / n_fraud, 4) if n_fraud else 0.0
        for tier in ("easy", "medium", "hard")
    }

    cardinality = {
        c: int(df[c].nunique(dropna=True))
        for c in (*ID_COLUMNS, *CATEGORICAL_FEATURES)
        if c in df.columns
    }

    numeric_summary = {
        c: _series_stats(df[c]) for c in NUMERIC_FEATURES if c in df.columns
    }

    # Class separation sanity: amount distribution by class.
    amt = pd.to_numeric(df["amount"], errors="coerce")
    amount_by_class = {
        "legit_p50": float(amt[~df["is_fraud"]].median()),
        "fraud_p50": float(amt[df["is_fraud"]].median()) if n_fraud else 0.0,
        "legit_mean": float(amt[~df["is_fraud"]].mean()),
        "fraud_mean": float(amt[df["is_fraud"]].mean()) if n_fraud else 0.0,
    }

    return {
        "n_rows": n,
        "n_columns": df.shape[1],
        "n_fraud": n_fraud,
        "n_legit": n_legit,
        "fraud_rate": round(n_fraud / n, 6) if n else 0.0,
        "imbalance_ratio": round(n_legit / n_fraud, 2) if n_fraud else float("inf"),
        "n_users": int(df["user_id"].nunique()),
        "fraud_per_tier": dict(tier_counts),
        "fraud_tier_share": tier_share,
        "fraud_type_counts": dict(fraud_type_counts),
        "n_fraud_scenarios_present": len(fraud_type_counts),
        "cardinality": cardinality,
        "numeric_summary": numeric_summary,
        "amount_by_class": amount_by_class,
        "duplicate_rows": int(df.drop(columns=["transaction_id"], errors="ignore").duplicated().sum()),
    }
