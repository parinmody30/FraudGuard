"""CTGAN-readiness assessment (analysis only — no synthesis is performed here).

CTGAN is a *row-independent* tabular synthesizer: it learns the joint distribution of a
single row's columns and samples rows i.i.d. This module judges how well the FraudLab
dataset fits that modeling assumption and emits concrete, actionable recommendations. It
does **not** implement or train CTGAN.

Dimensions assessed:
* Sample size — enough rows (and enough *minority* rows) to fit a GAN.
* Column types — mix of continuous / discrete CTGAN must be told about.
* Cardinality — high-cardinality identifier columns blow up the conditional one-hot space
  and must be dropped before training.
* Class imbalance — rare fraud needs conditional sampling, not naive sampling.
* Sequential signal — taken from the sequential-complexity analysis: row-wise synthesis
  cannot reproduce hard-tier fraud that only exists across ordered rows.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from fraudlab.evaluation.features import (
    CATEGORICAL_FEATURES,
    ID_COLUMNS,
    NUMERIC_FEATURES,
)

# Heuristic thresholds (widely-cited CTGAN practice).
MIN_ROWS_OK = 10_000
MIN_MINORITY_ROWS_OK = 500
HIGH_CARDINALITY = 50  # discrete columns above this are costly/leaky for CTGAN conditioning


def assess_ctgan_readiness(
    df: pd.DataFrame, sequential_findings: dict[str, Any]
) -> dict[str, Any]:
    """Produce a structured readiness verdict with per-dimension findings + recommendations."""
    n = len(df)
    n_fraud = int(df["is_fraud"].sum())

    findings: list[dict[str, Any]] = []
    recommendations: list[str] = []
    blockers: list[str] = []

    # --- sample size --------------------------------------------------------------
    size_ok = n >= MIN_ROWS_OK
    findings.append(
        {
            "dimension": "sample_size",
            "status": "ok" if size_ok else "marginal",
            "detail": f"{n} rows (CTGAN comfort >= {MIN_ROWS_OK}).",
        }
    )
    if not size_ok:
        recommendations.append(
            f"Generate >= {MIN_ROWS_OK} rows before CTGAN; current {n} risks underfitting."
        )

    # --- minority support ---------------------------------------------------------
    minority_ok = n_fraud >= MIN_MINORITY_ROWS_OK
    findings.append(
        {
            "dimension": "minority_support",
            "status": "ok" if minority_ok else "weak",
            "detail": f"{n_fraud} fraud rows (need >= {MIN_MINORITY_ROWS_OK} for stable minority modeling).",
        }
    )
    if not minority_ok:
        recommendations.append(
            "Too few fraud rows for the GAN to learn the minority mode; raise fraud_rate or "
            "scale up the dataset, and train CTGAN with conditional sampling on is_fraud."
        )

    # --- high-cardinality / identifier columns ------------------------------------
    high_card = {
        c: int(df[c].nunique())
        for c in (*ID_COLUMNS, *CATEGORICAL_FEATURES)
        if c in df.columns and df[c].nunique() > HIGH_CARDINALITY
    }
    findings.append(
        {
            "dimension": "high_cardinality_columns",
            "status": "action_required" if high_card else "ok",
            "detail": high_card or "no discrete column exceeds the cardinality budget.",
        }
    )
    drop_cols = sorted(set(ID_COLUMNS) | set(high_card))
    recommendations.append(
        "Drop identifier / high-cardinality columns before CTGAN and re-mint them "
        f"post-hoc: {drop_cols}."
    )

    # --- column typing for CTGAN --------------------------------------------------
    discrete_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    continuous_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    findings.append(
        {
            "dimension": "column_typing",
            "status": "ok",
            "detail": {
                "discrete_columns": discrete_cols,
                "continuous_columns": continuous_cols,
                "note": "velocity_1h/velocity_24h/account_age_days are integer-valued; pass "
                "as discrete or round CTGAN output.",
            },
        }
    )
    recommendations.append(
        "Declare discrete_columns explicitly to CTGAN; mode-specific normalization handles "
        "the skewed/multimodal 'amount' distribution."
    )

    # --- class imbalance ----------------------------------------------------------
    imbalance = round((n - n_fraud) / n_fraud, 1) if n_fraud else float("inf")
    findings.append(
        {
            "dimension": "class_imbalance",
            "status": "action_required",
            "detail": f"~{imbalance}:1 legit:fraud - naive sampling will under-represent fraud.",
        }
    )
    recommendations.append(
        "Use CTGAN conditional sampling (condition on is_fraud, and ideally on fraud_type) "
        "to control the synthetic fraud rate and per-scenario mix."
    )

    # --- sequential signal (the decisive dimension) -------------------------------
    hard_is_sequential = bool(sequential_findings.get("hard_is_sequential"))
    hard_lift = sequential_findings.get("recall_lift_from_sequence", {}).get("hard")
    findings.append(
        {
            "dimension": "sequential_structure",
            "status": "blocker_for_hard_tier" if hard_is_sequential else "ok",
            "detail": (
                "Hard-tier fraud is sequential: per-row recall is low and rises by "
                f"{hard_lift} when sequence-context features are added. A row-independent "
                "CTGAN samples rows i.i.d. and cannot reproduce this cross-row structure."
                if hard_is_sequential
                else "No strong sequential dependence detected; row-wise synthesis is adequate."
            ),
        }
    )
    if hard_is_sequential:
        blockers.append(
            "Row-wise CTGAN will not preserve hard-tier (sequential) fraud "
            "(gradual_escalation, money_mule, cross_channel, multi_stage, synthetic_identity)."
        )
        recommendations.append(
            "Scope CTGAN to per-row marginals (easy/medium fraud + legit realism). For "
            "hard/sequential fraud use a sequence-aware model (e.g. DoppelGANger / PAR / a "
            "time-series GAN over per-user sequences) or keep the rule-based simulator for "
            "those scenarios."
        )

    # --- overall verdict ----------------------------------------------------------
    if blockers:
        verdict = "ready_with_constraints"
        headline = (
            "Suitable for CTGAN as a per-row marginal/realism model, but NOT as a "
            "drop-in replacement for the simulator: it cannot reproduce sequential fraud."
        )
    elif not size_ok or not minority_ok:
        verdict = "prepare_first"
        headline = "Scale up rows / minority support before training CTGAN."
    else:
        verdict = "ready"
        headline = "Dataset is well-suited to row-wise CTGAN after standard preprocessing."

    return {
        "verdict": verdict,
        "headline": headline,
        "thresholds": {
            "min_rows": MIN_ROWS_OK,
            "min_minority_rows": MIN_MINORITY_ROWS_OK,
            "high_cardinality": HIGH_CARDINALITY,
        },
        "findings": findings,
        "blockers": blockers,
        "recommendations": recommendations,
        "preprocessing_drop_columns": drop_cols,
        "discrete_columns_for_ctgan": discrete_cols,
        "continuous_columns_for_ctgan": continuous_cols,
    }
