"""Tests for the Phase 3 fraud-detection benchmark suite.

These assert the contracts (group-split integrity, leakage-safe preprocessing, metric and
top-K correctness, leaderboard ordering, and the hybrid decision logic). A small simulator
dataset is generated once per module; the heavy end-to-end run lives in the CLI script.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudlab.config.config import BusinessType, FraudLabConfig, RiskAppetite
from fraudlab.exporters.export import transactions_to_dataframe
from fraudlab.models.benchmark import run_dataset_benchmark
from fraudlab.models.leaderboard import build_leaderboard, compute_improvement, decide
from fraudlab.models.metrics import compute_metrics, recall_at_fraction, tiered_recall
from fraudlab.models.preprocessing import (
    EXCLUDED_FEATURES,
    build_design,
    grouped_split,
    make_preprocessor,
)
from fraudlab.pipeline import build_dataset


@pytest.fixture(scope="module")
def bench_df() -> pd.DataFrame:
    config = FraudLabConfig(
        num_users=120,
        transactions_per_user=40,
        fraud_rate=0.08,
        business_type=BusinessType.BANK,
        risk_appetite=RiskAppetite.MEDIUM,
        random_seed=202,
    )
    transactions, _ = build_dataset(config)
    return transactions_to_dataframe(transactions)


# --------------------------------------------------------------- preprocessing / split

def test_design_excludes_leaky_columns(bench_df):
    design = build_design(bench_df)
    for col in (*EXCLUDED_FEATURES, "user_id", "timestamp", "is_fraud"):
        assert col not in design.X.columns
    # Derived causal temporal features are present.
    assert {"hour", "day_of_week", "is_weekend"} <= set(design.X.columns)


def test_group_split_no_user_overlap(bench_df):
    design = build_design(bench_df)
    train_idx, test_idx = grouped_split(design, test_size=0.2, random_state=42)
    train_users = set(design.groups[train_idx])
    test_users = set(design.groups[test_idx])
    assert train_users.isdisjoint(test_users)
    assert len(train_idx) + len(test_idx) == len(bench_df)


def test_preprocessing_consistency(bench_df):
    design = build_design(bench_df)
    train_idx, test_idx = grouped_split(design, random_state=42)
    pre = make_preprocessor(design)
    pre.fit(design.X.iloc[train_idx])
    Xtr = pre.transform(design.X.iloc[train_idx])
    Xte = pre.transform(design.X.iloc[test_idx])
    # Same column space across folds and no NaNs after imputation/encoding.
    assert Xtr.shape[1] == Xte.shape[1]
    assert not np.isnan(Xtr).any()
    assert not np.isnan(Xte).any()


# --------------------------------------------------------------------------- metrics

def test_recall_at_fraction_perfect_ranking():
    # Positives have the highest scores -> reviewing top 20% catches all 2 of 10 positives.
    scores = np.array([0.9, 0.8, 0.1, 0.2, 0.3, 0.15, 0.05, 0.25, 0.35, 0.4])
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    assert recall_at_fraction(scores, y, 0.2) == 1.0
    assert recall_at_fraction(scores, y, 0.1) == 0.5


def test_tiered_recall_counts():
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.2])
    y = np.array([1, 1, 1, 0, 0])
    tiers = np.array(["easy", "medium", "hard", "legit", "legit"], dtype=object)
    # Flag top-3 -> all three fraud tiers fully recalled.
    out = tiered_recall(scores, y, tiers, k=3)
    assert out == {"easy": 1.0, "medium": 1.0, "hard": 1.0}


def test_compute_metrics_keys_and_ranges():
    rng = np.random.default_rng(0)
    y = np.array([0] * 90 + [1] * 10)
    scores = np.concatenate([rng.uniform(0, 0.6, 90), rng.uniform(0.4, 1.0, 10)])
    labels = (scores >= 0.5).astype(int)
    tiers = np.array(["legit"] * 90 + ["easy"] * 4 + ["medium"] * 3 + ["hard"] * 3, dtype=object)
    m = compute_metrics(y, scores, labels, tiers)
    for key in ("roc_auc", "pr_auc", "precision", "recall", "f1",
                "recall_easy", "recall_medium", "recall_hard",
                "recall_at_1pct", "recall_at_2pct", "recall_at_5pct"):
        assert key in m
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["pr_auc"] <= 1.0


# ------------------------------------------------------------------------ leaderboard

def test_leaderboard_orders_by_pr_auc():
    results = {
        "a": {"pr_auc": 0.5, "recall_hard": 0.2},
        "b": {"pr_auc": 0.9, "recall_hard": 0.4},
        "c": {"pr_auc": 0.7, "recall_hard": 0.3},
        "skipme": {"skipped": True},
    }
    lb = build_leaderboard(results, sort_by="pr_auc")
    assert list(lb["model"]) == ["b", "c", "a"]
    assert list(lb["rank"]) == [1, 2, 3]
    assert "skipme" not in set(lb["model"])


def test_decide_outcomes():
    # Hybrid improves PR-AUC for most models -> A.
    sim = {"m1": {"pr_auc": 0.5, "recall_hard": 0.2}, "m2": {"pr_auc": 0.6, "recall_hard": 0.3}}
    hyb = {"m1": {"pr_auc": 0.7, "recall_hard": 0.2}, "m2": {"pr_auc": 0.8, "recall_hard": 0.3}}
    assert decide(compute_improvement(sim, hyb))["verdict"] == "A"

    # Approximately equal -> B.
    hyb_eq = {"m1": {"pr_auc": 0.505, "recall_hard": 0.2}, "m2": {"pr_auc": 0.602, "recall_hard": 0.3}}
    assert decide(compute_improvement(sim, hyb_eq))["verdict"] == "B"

    # Hybrid degrades PR-AUC and recall_hard for most -> C.
    hyb_bad = {"m1": {"pr_auc": 0.3, "recall_hard": 0.05}, "m2": {"pr_auc": 0.4, "recall_hard": 0.1}}
    assert decide(compute_improvement(sim, hyb_bad))["verdict"] == "C"


# ------------------------------------------------------------------ end-to-end (light)

def test_run_dataset_benchmark_smoke(bench_df):
    report = run_dataset_benchmark(
        bench_df, name="simulator", test_size=0.2, random_state=42, include_sequential=False
    )
    assert report["split"]["users_disjoint"] is True
    # Core supervised models + isolation forest are always present.
    assert {"logistic_regression", "random_forest", "gradient_boosting", "isolation_forest"} <= set(
        report["models"]
    )
    for m in report["models"].values():
        if not m.get("skipped"):
            assert 0.0 <= m["pr_auc"] <= 1.0
