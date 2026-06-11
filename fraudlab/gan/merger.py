"""Merge simulator hard fraud with CTGAN-generated soft rows into a hybrid dataset.

Final composition:
* **Simulator** hard-fraud rows (sequential scenarios) - taken verbatim from a simulator
  dataset, preserving each victim's multi-row sequence under a shared re-minted user id.
* **CTGAN** legitimate + easy-fraud + medium-fraud rows - per-row realism samples.

The requested fraud rate and per-scenario mix are honored across both sources. After
assembly all identifiers are freshly minted so the hybrid is internally consistent and
globally unique, matching the simulator's id formats (``T``/``U``/``M``/``D`` prefixes).
The exported schema, column order, and dtypes are unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fraudlab.config.config import ALL_FRAUD_TYPES, FRAUD_TIERS
from fraudlab.gan.dataset_builder import HARD_FRAUD_TYPES
from fraudlab.gan.sampler import CTGANSampler, _allocate
from fraudlab.models import TRANSACTION_FIELDS

_TYPE_TIER = {t: tier for tier, names in FRAUD_TIERS.items() for t in names}

# Identifier columns re-minted after sampling (merchant_name is re-derived from merchant_id).
REMINT_COLUMNS: tuple[str, ...] = (
    "transaction_id",
    "user_id",
    "merchant_id",
    "device_id",
    "merchant_name",
)


def build_hybrid_dataset(
    simulator_df: pd.DataFrame,
    sampler: CTGANSampler,
    *,
    n_rows: int,
    fraud_rate: float,
    fraud_mix: dict[str, float],
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assemble a hybrid dataset of ``n_rows`` with the requested fraud rate and mix.

    ``fraud_mix`` may span all 14 scenarios; hard-tier weights are satisfied from
    ``simulator_df`` and easy/medium weights from ``sampler``. Returns ``(df, composition)``.
    """
    rng = np.random.default_rng(seed)

    weights = {t: w for t, w in fraud_mix.items() if t in ALL_FRAUD_TYPES and w > 0}
    if not weights:
        raise ValueError("fraud_mix must assign positive weight to at least one scenario")

    n_fraud = int(round(n_rows * fraud_rate))
    n_fraud = max(0, min(n_rows, n_fraud))
    n_legit = n_rows - n_fraud
    per_type = _allocate(n_fraud, weights)

    hard_counts = {t: c for t, c in per_type.items() if _TYPE_TIER[t] == "hard"}
    soft_counts = {t: c for t, c in per_type.items() if _TYPE_TIER[t] in ("easy", "medium")}

    # --- hard fraud: sample rows from the simulator pool (with replacement if needed) ---
    hard_pool = simulator_df[simulator_df["fraud_type"].isin(HARD_FRAUD_TYPES)]
    hard_parts: list[pd.DataFrame] = []
    hard_realized: dict[str, int] = {}
    for ftype, count in hard_counts.items():
        if count <= 0:
            continue
        pool_t = hard_pool[hard_pool["fraud_type"] == ftype]
        if len(pool_t) == 0:
            continue
        idx = rng.integers(0, len(pool_t), size=count)
        part = pool_t.iloc[idx].copy()
        part["_source"] = "simulator_hard"
        hard_parts.append(part)
        hard_realized[ftype] = count

    # --- soft fraud + legit: CTGAN ---
    soft_parts: list[pd.DataFrame] = []
    soft_realized: dict[str, int] = {}
    for i, ftype in enumerate(sorted(soft_counts)):
        count = soft_counts[ftype]
        if count <= 0:
            continue
        part = sampler.conditional_generate(ftype, count, seed=seed + 300 + i)
        part["_source"] = "ctgan_soft"
        soft_parts.append(part)
        soft_realized[ftype] = int(len(part))

    legit_part = None
    if n_legit > 0:
        legit_part = sampler.conditional_generate(None, n_legit, seed=seed + 17)
        legit_part["_source"] = "ctgan_legit"

    parts = [*hard_parts, *soft_parts]
    if legit_part is not None:
        parts.append(legit_part)
    if not parts:
        raise ValueError("nothing to merge (check n_rows / fraud_rate / fraud_mix)")

    # Union to the full schema (CTGAN parts lack the dropped id columns -> NaN for now).
    aligned = [p.reindex(columns=[*TRANSACTION_FIELDS, "_source"]) for p in parts]
    df = pd.concat(aligned, ignore_index=True)

    # Interleave sources deterministically before re-minting.
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)

    df = _remint_identifiers(df, rng)
    df = _coerce_schema(df)

    composition = _composition(df, hard_realized, soft_realized, n_legit, n_rows, fraud_rate)
    df = df.drop(columns=["_source"], errors="ignore")
    return df[list(TRANSACTION_FIELDS)], composition


def _remint_identifiers(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Freshly mint all identifiers, preserving per-user grouping for simulator rows.

    Simulator hard rows that shared an original ``user_id`` keep a shared new ``user_id``
    (so each sequential-fraud episode stays intact). CTGAN rows are independent and each
    receives its own new user. Merchants are drawn from a synthetic shared pool so merchant
    cardinality stays realistic; ``transaction_id`` is globally sequential.
    """
    n = len(df)

    # Synthetic merchant pool (shared) -> realistic merchant_id reuse.
    pool_size = max(1, round(n / 40))
    merch_ids = [f"M{i:06d}" for i in range(pool_size)]
    merch_assign = rng.integers(0, pool_size, size=n)

    sources = df["_source"].to_numpy()
    orig_users = df["user_id"].to_numpy()

    new_user = np.empty(n, dtype=object)
    new_device = np.empty(n, dtype=object)
    new_merchant = np.empty(n, dtype=object)
    new_name = np.empty(n, dtype=object)
    new_txn = np.empty(n, dtype=object)

    user_map: dict[str, str] = {}
    user_counter = 0

    for i in range(n):
        src = sources[i]
        if src == "simulator_hard" and isinstance(orig_users[i], str):
            key = orig_users[i]
            uid = user_map.get(key)
            if uid is None:
                uid = f"U{user_counter:07d}"
                user_map[key] = uid
                user_counter += 1
        else:
            uid = f"U{user_counter:07d}"
            user_counter += 1
        new_user[i] = uid
        new_device[i] = f"D{uid[1:]}00"
        mid = merch_ids[merch_assign[i]]
        new_merchant[i] = mid
        new_name[i] = f"Merchant {mid}"
        new_txn[i] = f"T{i:010d}"

    df["transaction_id"] = new_txn
    df["user_id"] = new_user
    df["device_id"] = new_device
    df["merchant_id"] = new_merchant
    df["merchant_name"] = new_name
    return df


def _coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce the canonical column dtypes (mirrors the exporter's coercion)."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for c in ("account_age_days", "velocity_1h", "velocity_24h"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).round().astype("int64")
    for c in ("amount", "avg_user_spend", "median_user_spend", "merchant_risk_score", "device_risk_score"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["is_foreign"] = df["is_foreign"].astype("bool")
    df["is_fraud"] = df["is_fraud"].astype("bool")
    df["fraud_type"] = df["fraud_type"].where(df["fraud_type"].notna(), None)
    if "currency" in df:
        df["currency"] = df["currency"].astype(str)
    return df


def _composition(
    df: pd.DataFrame,
    hard_realized: dict[str, int],
    soft_realized: dict[str, int],
    n_legit: int,
    n_rows: int,
    fraud_rate: float,
) -> dict[str, Any]:
    realized_fraud = int(df["is_fraud"].sum())
    by_tier = {tier: 0 for tier in ("easy", "medium", "hard")}
    for ftype, cnt in {**hard_realized, **soft_realized}.items():
        by_tier[_TYPE_TIER[ftype]] += cnt
    return {
        "n_rows": int(len(df)),
        "requested_rows": n_rows,
        "requested_fraud_rate": fraud_rate,
        "realized_fraud_rate": round(realized_fraud / len(df), 4) if len(df) else 0.0,
        "n_legit": int((~df["is_fraud"]).sum()),
        "n_fraud": realized_fraud,
        "fraud_by_tier": by_tier,
        "hard_from_simulator": hard_realized,
        "soft_from_ctgan": soft_realized,
        "legit_from_ctgan": int(n_legit),
        "unique_transaction_ids": int(df["transaction_id"].nunique()),
        "unique_users": int(df["user_id"].nunique()),
        "source_breakdown": df["_source"].value_counts().to_dict() if "_source" in df else {},
    }
