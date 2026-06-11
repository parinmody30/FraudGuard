"""Reusable, leakage-safe preprocessing for the fraud-detection benchmark (Phase 3).

Responsibilities:
* Derive causal temporal features from ``timestamp`` (hour / day-of-week / weekend).
* Classify columns into continuous / categorical / boolean roles.
* Exclude identifier and label-proxy columns so detectors cannot cheat.
* Provide a scikit-learn ``ColumnTransformer`` (median-impute + scale; one-hot; binary)
  and a ``GroupShuffleSplit`` keyed on ``user_id`` so no user spans train and test.

This module is imported as ``fraudlab.models.preprocessing``; it adds a new module to the
``fraudlab.models`` package without touching the existing entity dataclasses or their
package ``__init__`` (so ``from fraudlab.models import Transaction`` is unaffected).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraudlab.config.config import FRAUD_TIERS

# Columns that must never become features (identifiers + the label proxy). The label
# itself (is_fraud) and the grouping key (user_id) are handled separately.
EXCLUDED_FEATURES: tuple[str, ...] = (
    "fraud_type",
    "transaction_id",
    "merchant_id",
    "merchant_name",
    "device_id",
)
LABEL_COLUMN = "is_fraud"
GROUP_COLUMN = "user_id"
TIMESTAMP_COLUMN = "timestamp"

CONTINUOUS_FEATURES: tuple[str, ...] = (
    "amount",
    "account_age_days",
    "avg_user_spend",
    "median_user_spend",
    "velocity_1h",
    "velocity_24h",
    "merchant_risk_score",
    "device_risk_score",
    "hour",
)
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "currency",
    "merchant_category",
    "payment_method",
    "channel",
    "device_type",
    "city",
    "state",
    "country",
    "day_of_week",
)
BOOLEAN_FEATURES: tuple[str, ...] = ("is_foreign", "is_weekend")

_TIER_OF_TYPE = {name: tier for tier, names in FRAUD_TIERS.items() for name in names}


def _tier_of(fraud_type: object) -> str:
    if fraud_type is None or (isinstance(fraud_type, float) and np.isnan(fraud_type)):
        return "legit"
    return _TIER_OF_TYPE.get(str(fraud_type), "unknown")


@dataclass(slots=True)
class Design:
    """Model-ready design: raw (un-encoded) feature frame + aligned label/group/tier arrays."""

    X: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    tiers: np.ndarray
    fraud_types: np.ndarray
    timestamps: np.ndarray
    continuous: list[str]
    categorical: list[str]
    boolean: list[str]


def build_design(df: pd.DataFrame) -> Design:
    """Engineer features and split out label/group/tier arrays from a raw dataset.

    Temporal features are derived causally from each row's own timestamp; no excluded
    identifier/label-proxy column is carried into ``X``.
    """
    ts = pd.to_datetime(df[TIMESTAMP_COLUMN])
    work = pd.DataFrame(index=df.index)

    for c in CONTINUOUS_FEATURES:
        if c == "hour":
            work["hour"] = ts.dt.hour.astype("float64")
        else:
            work[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    for c in CATEGORICAL_FEATURES:
        if c == "day_of_week":
            work["day_of_week"] = ts.dt.dayofweek.astype("int64").astype(str)
        else:
            work[c] = df[c].astype(str)

    work["is_foreign"] = df["is_foreign"].astype(bool).astype("int8")
    work["is_weekend"] = (ts.dt.dayofweek >= 5).astype("int8")

    y = df[LABEL_COLUMN].astype(int).to_numpy()
    groups = df[GROUP_COLUMN].astype(str).to_numpy()
    ftypes = df["fraud_type"].to_numpy() if "fraud_type" in df else np.array([None] * len(df))
    tiers = np.array([_tier_of(ft) for ft in ftypes], dtype=object)

    return Design(
        X=work,
        y=y,
        groups=groups,
        tiers=tiers,
        fraud_types=ftypes,
        timestamps=ts.to_numpy(),
        continuous=list(CONTINUOUS_FEATURES),
        categorical=list(CATEGORICAL_FEATURES),
        boolean=list(BOOLEAN_FEATURES),
    )


def make_preprocessor(design: Design):
    """Build a ``ColumnTransformer`` implementing the required per-role transforms."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    continuous = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    boolean = SimpleImputer(strategy="most_frequent")  # already 0/1 -> binary passthrough

    return ColumnTransformer(
        transformers=[
            ("continuous", continuous, design.continuous),
            ("categorical", categorical, design.categorical),
            ("boolean", boolean, design.boolean),
        ],
        remainder="drop",
    )


def grouped_split(
    design: Design, *, test_size: float = 0.2, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Return train/test row indices with users disjoint across folds (GroupShuffleSplit)."""
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(design.X, design.y, groups=design.groups))
    return train_idx, test_idx
