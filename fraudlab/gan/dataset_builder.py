"""Build the CTGAN training set and convert between the schema frame and the model frame.

Training-set policy (Phase 2 scope):

* INCLUDE legitimate rows, easy-fraud rows, medium-fraud rows.
* EXCLUDE hard-fraud rows (the five sequential scenarios) - those stay simulator-only.
* DROP the five identifier columns before training (re-minted post-sampling).
* PRESERVE ``is_fraud`` and ``fraud_type`` so sampling can be conditioned on them.

``fraud_type`` is ``None`` on legitimate rows; CTGAN needs every discrete column non-null,
so legit rows are tagged with :data:`LEGIT_SENTINEL` in the model representation and mapped
back to ``None`` on decode. This is an internal encoding only - the exported schema is
unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from fraudlab.config.config import FRAUD_TIERS
from fraudlab.gan.metadata import DROP_COLUMNS, GANMetadata

# Hard (sequential) scenarios excluded from CTGAN training.
HARD_FRAUD_TYPES: tuple[str, ...] = tuple(FRAUD_TIERS["hard"])
# Easy+medium scenarios CTGAN is allowed to learn.
SOFT_FRAUD_TYPES: tuple[str, ...] = (*FRAUD_TIERS["easy"], *FRAUD_TIERS["medium"])

# Sentinel category standing in for "no fraud_type" (legitimate rows) during modelling.
LEGIT_SENTINEL = "legit"

# Reasonable epoch bounds (seconds) used to clamp decoded timestamps: 2000-01-01 .. 2100.
_EPOCH_MIN = int(datetime(2000, 1, 1, tzinfo=timezone.utc).timestamp())
_EPOCH_MAX = int(datetime(2100, 1, 1, tzinfo=timezone.utc).timestamp())


def build_training_frame(
    df: pd.DataFrame, *, drop_columns: tuple[str, ...] = DROP_COLUMNS
) -> pd.DataFrame:
    """Return the CTGAN training frame: soft rows only, identifiers dropped, sentinel-tagged.

    Input is left unmodified. ``fraud_type`` keeps easy/medium scenario names on fraud rows
    and :data:`LEGIT_SENTINEL` on legitimate rows.
    """
    ftype = df["fraud_type"]
    is_hard = ftype.isin(HARD_FRAUD_TYPES)
    kept = df.loc[~is_hard].copy()

    kept["fraud_type"] = kept["fraud_type"].where(kept["fraud_type"].notna(), LEGIT_SENTINEL)
    # Defensive: any non-soft, non-sentinel value would be an unexpected label.
    allowed = set(SOFT_FRAUD_TYPES) | {LEGIT_SENTINEL}
    kept = kept[kept["fraud_type"].isin(allowed)]

    kept = kept.drop(columns=[c for c in drop_columns if c in kept.columns])
    return kept.reset_index(drop=True)


def training_composition(df: pd.DataFrame) -> dict:
    """Summarize what a raw dataset contributes to / excludes from CTGAN training."""
    ftype = df["fraud_type"]
    is_hard = ftype.isin(HARD_FRAUD_TYPES)
    train = df.loc[~is_hard]
    return {
        "total_rows": int(len(df)),
        "training_rows": int(len(train)),
        "legit_rows": int((~train["is_fraud"]).sum()),
        "easy_fraud_rows": int(train["fraud_type"].isin(FRAUD_TIERS["easy"]).sum()),
        "medium_fraud_rows": int(train["fraud_type"].isin(FRAUD_TIERS["medium"]).sum()),
        "excluded_hard_rows": int(is_hard.sum()),
        "soft_fraud_types": list(SOFT_FRAUD_TYPES),
        "excluded_hard_types": list(HARD_FRAUD_TYPES),
    }


def to_model_frame(df: pd.DataFrame, metadata: GANMetadata) -> pd.DataFrame:
    """Encode a training frame into the all-numeric/string frame CTGAN consumes.

    Datetime -> epoch seconds (continuous); booleans -> 'true'/'false' strings; categoricals
    -> string. Column order follows ``metadata.columns``.
    """
    out = pd.DataFrame(index=df.index)
    for c in metadata.columns:
        if c in metadata.datetime:
            ts = pd.to_datetime(df[c], utc=True, errors="coerce")
            out[c] = (ts.astype("int64") / 1e9).astype("float64")
        elif c in metadata.boolean:
            out[c] = df[c].astype(bool).map({True: "true", False: "false"})
        elif c in metadata.categorical:
            out[c] = df[c].astype("string").fillna(LEGIT_SENTINEL).astype(str)
        else:  # continuous
            out[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return out[metadata.columns]


def from_model_frame(df: pd.DataFrame, metadata: GANMetadata) -> pd.DataFrame:
    """Decode a CTGAN-sampled model frame back to schema-typed columns.

    Inverse of :func:`to_model_frame`, plus range/dtype hygiene: integer columns are
    rounded and clipped to >= 0, unit-interval columns clipped to [0, 1], ``amount`` clipped
    to >= 1 and rounded, the sentinel ``fraud_type`` mapped back to ``None`` on legit rows,
    and ``is_fraud`` made consistent with ``fraud_type``.
    """
    out = pd.DataFrame(index=df.index)
    for c in metadata.columns:
        if c in metadata.datetime:
            epoch = pd.to_numeric(df[c], errors="coerce").fillna(_EPOCH_MIN)
            epoch = epoch.clip(lower=_EPOCH_MIN, upper=_EPOCH_MAX)
            out[c] = pd.to_datetime(epoch, unit="s", utc=True).dt.tz_localize(None)
        elif c in metadata.boolean:
            out[c] = df[c].astype(str).str.lower().map(
                {"true": True, "false": False}
            ).fillna(False).astype(bool)
        elif c in metadata.categorical:
            out[c] = df[c].astype(str)
        else:  # continuous
            vals = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            if c in metadata.integer_columns:
                vals = vals.round().clip(lower=0).astype("int64")
            elif c in metadata.unit_interval_columns:
                vals = vals.clip(lower=0.0, upper=1.0).round(4)
            elif c == "amount":
                vals = vals.clip(lower=1.0).round(2)
            else:
                vals = vals.round(2)
            out[c] = vals

    # --- label hygiene ---------------------------------------------------------
    if "fraud_type" in out.columns:
        is_legit = out["fraud_type"].eq(LEGIT_SENTINEL)
        out["fraud_type"] = out["fraud_type"].where(~is_legit, other=None)
        if "is_fraud" in out.columns:
            # is_fraud is authoritative-by-construction: fraud iff a real fraud_type.
            out["is_fraud"] = out["fraud_type"].notna()
    return out


def decode_epoch_bounds() -> tuple[int, int]:
    """Expose the epoch clamp bounds (used by tests)."""
    return _EPOCH_MIN, _EPOCH_MAX
