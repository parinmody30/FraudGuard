"""Feature engineering for evaluation: encoding, splitting, and tier mapping.

The evaluator treats the exported transaction schema as its only contract. It classifies
columns into roles (identifier / label / leaky-by-construction / categorical / numeric),
builds a model-ready feature matrix using only *legitimately usable* causal features, and
provides a user-grouped train/test split so no user spans both folds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraudlab.config.config import FRAUD_TIERS

# Columns that identify a row/entity and must never be used as predictive features
# (they would let a model memorize specific users/merchants/devices).
ID_COLUMNS: tuple[str, ...] = (
    "transaction_id",
    "user_id",
    "merchant_id",
    "merchant_name",
    "device_id",
)

# The label, and the column that is non-null *only* for fraud rows. ``fraud_type`` is a
# perfect proxy for the label by construction and is excluded from features (the leakage
# module verifies this explicitly rather than assuming it).
LABEL_COLUMN = "is_fraud"
LEAKY_BY_CONSTRUCTION: tuple[str, ...] = ("fraud_type",)

# Raw timestamp is decomposed into causal cyclical features instead of being used directly.
TIMESTAMP_COLUMN = "timestamp"

# Low/medium-cardinality categoricals usable as features.
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "currency",
    "merchant_category",
    "payment_method",
    "channel",
    "device_type",
    "city",
    "state",
    "country",
    "is_foreign",
)

# Per-row numeric features. ``avg/median_user_spend`` and ``velocity_*`` are the causal
# rolling features computed as-of each transaction by the generator.
NUMERIC_FEATURES: tuple[str, ...] = (
    "amount",
    "account_age_days",
    "avg_user_spend",
    "median_user_spend",
    "velocity_1h",
    "velocity_24h",
    "merchant_risk_score",
    "device_risk_score",
)

# Temporal features derived causally from the timestamp (known at scoring time).
DERIVED_TEMPORAL_FEATURES: tuple[str, ...] = ("hour", "day_of_week", "is_weekend")

# Engineered per-row ratios that are legitimate (use only as-of values).
DERIVED_RATIO_FEATURES: tuple[str, ...] = ("amount_to_avg", "amount_to_median")


def fraud_type_to_tier() -> dict[str, str]:
    """Map each fraud scenario name to its difficulty tier (easy/medium/hard)."""
    return {name: tier for tier, names in FRAUD_TIERS.items() for name in names}


def tier_of_row(fraud_type: object, mapping: dict[str, str]) -> str:
    """Return the tier label for a row: 'legit' for non-fraud, else the scenario's tier."""
    if fraud_type is None or (isinstance(fraud_type, float) and np.isnan(fraud_type)):
        return "legit"
    return mapping.get(str(fraud_type), "unknown")


@dataclass(slots=True)
class FeatureBundle:
    """Model-ready features plus the metadata an evaluator needs alongside them."""

    X: pd.DataFrame  # encoded numeric matrix (categoricals ordinal-coded)
    y: np.ndarray  # int label (1 = fraud)
    groups: np.ndarray  # user_id per row (for grouped splitting)
    tiers: np.ndarray  # 'legit' | 'easy' | 'medium' | 'hard' per row
    fraud_types: np.ndarray  # raw fraud_type per row (object)
    numeric_columns: list[str]
    categorical_columns: list[str]


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with causal temporal/ratio features added.

    All derived features use only information available at the transaction's own time, so
    they introduce no future leakage.
    """
    out = df.copy()
    ts = pd.to_datetime(out[TIMESTAMP_COLUMN])
    out["hour"] = ts.dt.hour.astype("int64")
    out["day_of_week"] = ts.dt.dayofweek.astype("int64")
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype("int64")

    avg = out["avg_user_spend"].replace(0, np.nan)
    med = out["median_user_spend"].replace(0, np.nan)
    out["amount_to_avg"] = (out["amount"] / avg).fillna(1.0).clip(upper=1e6)
    out["amount_to_median"] = (out["amount"] / med).fillna(1.0).clip(upper=1e6)
    return out


def build_feature_bundle(df: pd.DataFrame) -> FeatureBundle:
    """Construct an encoded feature matrix and aligned metadata from a raw dataset."""
    enriched = add_derived_columns(df)

    numeric_cols = [
        c
        for c in (*NUMERIC_FEATURES, *DERIVED_TEMPORAL_FEATURES, *DERIVED_RATIO_FEATURES)
        if c in enriched.columns
    ]
    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in enriched.columns]

    encoded = {}
    for c in numeric_cols:
        encoded[c] = pd.to_numeric(enriched[c], errors="coerce").astype("float64")
    for c in categorical_cols:
        # Stable ordinal coding: deterministic category order -> reproducible codes.
        codes = enriched[c].astype("category")
        encoded[c] = codes.cat.codes.astype("int64")

    X = pd.DataFrame(encoded, index=enriched.index)
    X = X.fillna(0.0)

    y = enriched[LABEL_COLUMN].astype(int).to_numpy()
    groups = enriched["user_id"].to_numpy()

    mapping = fraud_type_to_tier()
    ftypes = enriched["fraud_type"].to_numpy() if "fraud_type" in enriched else np.array([None] * len(enriched))
    tiers = np.array([tier_of_row(ft, mapping) for ft in ftypes], dtype=object)

    return FeatureBundle(
        X=X,
        y=y,
        groups=groups,
        tiers=tiers,
        fraud_types=ftypes,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
    )


def grouped_split(
    bundle: FeatureBundle, *, test_size: float = 0.3, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Return train/test row-index arrays with users disjoint across folds.

    Splitting by ``user_id`` is the realistic protocol for transaction fraud: a deployed
    detector scores users it has never seen, so letting a user's rows appear in both folds
    would inflate scores via per-user memorization.
    """
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(bundle.X, bundle.y, groups=bundle.groups))
    return train_idx, test_idx
