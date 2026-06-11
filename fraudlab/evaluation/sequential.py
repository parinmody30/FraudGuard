"""Sequential-complexity analysis.

Hard-tier fraud (gradual_escalation, money_mule, cross_channel, multi_stage,
synthetic_identity) is engineered so that no single row is anomalous — the signal lives in
the *ordered sequence* of a user's transactions. This module quantifies that.

Method: build a second feature set that adds **causal per-user sequence-context features**
(computed using only each row's own and prior rows, ordered by time), retrain the strong
row-wise learner, and measure the recall *lift* per tier versus the per-row-only features
from the benchmark. A large hard-tier lift from sequence aggregation — combined with low
hard-tier recall without it — demonstrates the fraud is sequential, not per-row.

The same finding drives the CTGAN recommendation: a row-independent synthesizer cannot
reproduce signal that only exists across ordered rows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fraudlab.evaluation.benchmark import _recall_at_k_by_tier
from fraudlab.evaluation.features import (
    add_derived_columns,
    build_feature_bundle,
    grouped_split,
)

# Trailing window (number of prior transactions, inclusive of current) for sequence stats.
SEQ_WINDOW = 8


def build_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute causal per-user sequence-context features, indexed like ``df``.

    Every feature at row *i* uses only transactions of the same user at positions <= i in
    time order, so it is leakage-free and computable online at scoring time.
    """
    enriched = add_derived_columns(df)
    enriched = enriched.assign(_orig_index=np.arange(len(enriched)))
    enriched = enriched.sort_values(["user_id", "timestamp"], kind="stable")

    out = pd.DataFrame(index=enriched.index)
    g = enriched.groupby("user_id", sort=False)

    amt = enriched["amount"].astype(float)
    # Expanding/rolling stats over the trailing window (current row included).
    roll_mean = g["amount"].transform(
        lambda s: s.rolling(SEQ_WINDOW, min_periods=1).mean()
    )
    roll_std = g["amount"].transform(
        lambda s: s.rolling(SEQ_WINDOW, min_periods=1).std().fillna(0.0)
    )
    # Z-score of the current amount vs its own trailing window: spikes & escalation.
    out["seq_amount_z"] = ((amt - roll_mean) / roll_std.replace(0, np.nan)).fillna(0.0)

    # Monotone-trend signal: Spearman-like rank slope of the last window of amounts.
    out["seq_amount_trend"] = g["amount"].transform(_trailing_trend)

    # Time gap (hours) since this user's previous transaction; bursts -> small gaps.
    ts = pd.to_datetime(enriched["timestamp"]).astype("int64") / 1e9
    out["seq_gap_hours"] = (
        ts.groupby(enriched["user_id"], sort=False).diff().fillna(1e6) / 3600.0
    ).clip(upper=1e4)

    # Rolling diversity of payment rails / channels: cross-channel coordination.
    out["seq_pm_diversity"] = g["payment_method"].transform(
        lambda s: _rolling_nunique(s, SEQ_WINDOW)
    )
    out["seq_channel_diversity"] = g["channel"].transform(
        lambda s: _rolling_nunique(s, SEQ_WINDOW)
    )

    # Is this user's merchant new relative to everything they've done before? fan-out/setup.
    out["seq_new_merchant"] = g["merchant_id"].transform(_is_first_occurrence)

    # Cumulative count of distinct devices seen so far: synthetic/new-device staging.
    out["seq_device_run"] = g["device_id"].transform(_expanding_nunique)

    # Restore original row order.
    out["_orig_index"] = enriched["_orig_index"].to_numpy()
    out = out.sort_values("_orig_index").drop(columns="_orig_index")
    out.index = df.index
    return out.astype("float64")


def _trailing_trend(s: pd.Series) -> pd.Series:
    """Rank-correlation of value vs position over the trailing window (in [-1, 1])."""
    vals = s.to_numpy(dtype=float)
    n = len(vals)
    res = np.zeros(n)
    for i in range(n):
        lo = max(0, i - SEQ_WINDOW + 1)
        window = vals[lo : i + 1]
        m = len(window)
        if m < 3 or np.all(window == window[0]):
            res[i] = 0.0
            continue
        ranks = pd.Series(window).rank().to_numpy()
        pos = np.arange(m, dtype=float)
        rc = np.corrcoef(ranks, pos)[0, 1]
        res[i] = 0.0 if np.isnan(rc) else rc
    return pd.Series(res, index=s.index)


def _rolling_nunique(s: pd.Series, window: int) -> pd.Series:
    codes = s.astype("category").cat.codes.to_numpy()
    n = len(codes)
    res = np.zeros(n)
    for i in range(n):
        lo = max(0, i - window + 1)
        res[i] = len(np.unique(codes[lo : i + 1]))
    return pd.Series(res, index=s.index)


def _is_first_occurrence(s: pd.Series) -> pd.Series:
    seen: set = set()
    res = np.zeros(len(s))
    for j, v in enumerate(s.to_numpy()):
        res[j] = 0.0 if v in seen else 1.0
        seen.add(v)
    return pd.Series(res, index=s.index)


def _expanding_nunique(s: pd.Series) -> pd.Series:
    seen: set = set()
    res = np.zeros(len(s))
    for j, v in enumerate(s.to_numpy()):
        seen.add(v)
        res[j] = len(seen)
    return pd.Series(res, index=s.index)


def analyze_sequential_complexity(
    df: pd.DataFrame, *, test_size: float = 0.3, random_state: int = 42
) -> dict[str, Any]:
    """Quantify how much per-tier recall improves when sequence context is added."""
    from sklearn.ensemble import GradientBoostingClassifier

    bundle = build_feature_bundle(df)
    train_idx, test_idx = grouped_split(
        bundle, test_size=test_size, random_state=random_state
    )
    y = bundle.y
    tiers = bundle.tiers
    y_test = y[test_idx]
    tiers_test = tiers[test_idx]
    k = int((y_test == 1).sum())

    seq = build_sequence_features(df)
    seq_columns = list(seq.columns)

    base_X = bundle.X
    aug_X = pd.concat([bundle.X, seq], axis=1)

    def _fit_score(X: pd.DataFrame) -> dict[str, Any]:
        clf = GradientBoostingClassifier(random_state=random_state)
        clf.fit(X.iloc[train_idx], y[train_idx])
        scores = clf.predict_proba(X.iloc[test_idx])[:, list(clf.classes_).index(1)]
        rec = _recall_at_k_by_tier(scores, y_test, tiers_test, k)
        rec.pop("_flagged", None)
        return rec

    base_recall = _fit_score(base_X)
    aug_recall = _fit_score(aug_X)

    def _lift(tier: str) -> float | None:
        b, a = base_recall.get(tier), aug_recall.get(tier)
        if b is None or a is None:
            return None
        return round(a - b, 4)

    lift = {tier: _lift(tier) for tier in ("easy", "medium", "hard", "overall")}

    # How separable is each tier on the sequence features alone? (univariate evidence)
    seq_separation = _tier_separation(df, seq)

    return {
        "window": SEQ_WINDOW,
        "sequence_features": seq_columns,
        "per_row_recall": {t: base_recall.get(t) for t in ("easy", "medium", "hard", "overall")},
        "sequence_augmented_recall": {
            t: aug_recall.get(t) for t in ("easy", "medium", "hard", "overall")
        },
        "recall_lift_from_sequence": lift,
        "tier_separation_on_sequence_features": seq_separation,
        "hard_is_sequential": bool(
            (lift.get("hard") or 0) >= 0.05
            and (base_recall.get("hard") or 0) < (base_recall.get("easy") or 1)
        ),
    }


def _tier_separation(df: pd.DataFrame, seq: pd.DataFrame) -> dict[str, float]:
    """Mean per-feature univariate AUC of sequence features for each fraud tier vs legit."""
    from sklearn.metrics import roc_auc_score

    from fraudlab.evaluation.features import fraud_type_to_tier, tier_of_row

    mapping = fraud_type_to_tier()
    row_tiers = np.array(
        [tier_of_row(ft, mapping) for ft in df.get("fraud_type", pd.Series([None] * len(df)))]
    )
    legit = row_tiers == "legit"
    result: dict[str, float] = {}
    for tier in ("easy", "medium", "hard"):
        mask = legit | (row_tiers == tier)
        sub = seq[mask]
        y = (row_tiers[mask] == tier).astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            result[tier] = float("nan")
            continue
        aucs = []
        for col in seq.columns:
            x = sub[col].to_numpy(dtype=float)
            if np.all(x == x[0]):
                continue
            a = roc_auc_score(y, x)
            aucs.append(max(a, 1.0 - a))
        result[tier] = round(float(np.mean(aucs)), 4) if aucs else float("nan")
    return result
