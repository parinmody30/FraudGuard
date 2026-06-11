"""Tests for the Phase 1.1 dataset evaluation framework.

These verify the evaluator's contracts (structure, determinism, leakage-cleanliness,
grouped-split integrity, and sequential-signal detection) without asserting exact model
scores, which can vary slightly across sklearn/BLAS builds.
"""

from __future__ import annotations

import pytest

from fraudlab.config.config import BusinessType, FraudLabConfig, RiskAppetite
from fraudlab.evaluation import evaluate_dataset, format_report
from fraudlab.evaluation.features import (
    LEAKY_BY_CONSTRUCTION,
    build_feature_bundle,
    grouped_split,
)
from fraudlab.evaluation.sequential import build_sequence_features
from fraudlab.exporters.export import transactions_to_dataframe
from fraudlab.pipeline import build_dataset


@pytest.fixture(scope="module")
def eval_df():
    config = FraudLabConfig(
        num_users=120,
        transactions_per_user=50,
        fraud_rate=0.06,
        business_type=BusinessType.BANK,
        risk_appetite=RiskAppetite.MEDIUM,
        random_seed=2024,
    )
    transactions, _ = build_dataset(config)
    return transactions_to_dataframe(transactions)


@pytest.fixture(scope="module")
def report(eval_df):
    return evaluate_dataset(eval_df, test_size=0.3, random_state=42)


def test_report_has_all_sections(report):
    assert set(report) == {"metrics", "benchmark", "leakage", "sequential", "ctgan_readiness"}


def test_metrics_basic(report):
    m = report["metrics"]
    assert m["n_rows"] == m["n_fraud"] + m["n_legit"]
    assert 0.0 < m["fraud_rate"] < 0.2
    assert m["n_fraud_scenarios_present"] >= 1
    # Generator guarantees globally unique transaction ids -> no duplicate content rows.
    assert m["duplicate_rows"] == 0


def test_features_exclude_ids_and_label(eval_df):
    bundle = build_feature_bundle(eval_df)
    cols = set(bundle.X.columns)
    for forbidden in ("transaction_id", "user_id", "is_fraud", *LEAKY_BY_CONSTRUCTION):
        assert forbidden not in cols


def test_grouped_split_users_disjoint(eval_df):
    bundle = build_feature_bundle(eval_df)
    train_idx, test_idx = grouped_split(bundle, random_state=7)
    train_users = set(bundle.groups[train_idx])
    test_users = set(bundle.groups[test_idx])
    assert train_users.isdisjoint(test_users)
    assert len(train_idx) + len(test_idx) == len(bundle.X)


def test_construction_leakage_flagged_and_excluded(report):
    lk = report["leakage"]
    ftype = next(c for c in lk["construction_leakage"] if c["column"] == "fraud_type")
    assert ftype["leaks_label"] is True
    assert ftype["excluded_from_features"] is True


def test_no_single_feature_leakage(report):
    # A realistic dataset should have no single *usable* feature that nearly perfectly
    # separates the classes; difficulty must come from signal structure, not a giveaway.
    assert report["leakage"]["clean"] is True
    assert report["leakage"]["single_feature_leakage"] == []


def test_benchmark_detectors_present(report):
    dets = report["benchmark"]["detectors"]
    assert set(dets) == {"rule_baseline", "logistic", "gradient_boosting"}
    assert report["benchmark"]["split"]["users_disjoint"] is True
    for r in dets.values():
        assert 0.0 <= r["recall_overall"] <= 1.0


def test_learned_model_beats_rule_baseline(report):
    dets = report["benchmark"]["detectors"]
    # The trained learner should rank fraud at least as well as the naive per-row rules.
    assert dets["gradient_boosting"]["pr_auc"] >= dets["rule_baseline"]["pr_auc"]


def test_sequence_features_are_causal_shape(eval_df):
    seq = build_sequence_features(eval_df)
    assert len(seq) == len(eval_df)
    assert list(seq.index) == list(eval_df.index)
    assert seq.isna().sum().sum() == 0


def test_sequential_section_quantifies_lift(report):
    sq = report["sequential"]
    assert "recall_lift_from_sequence" in sq
    assert set(sq["per_row_recall"]) == {"easy", "medium", "hard", "overall"}
    # Adding causal sequence context should not hurt overall recall.
    lift = sq["recall_lift_from_sequence"]["overall"]
    assert lift is None or lift >= -0.05


def test_ctgan_readiness_verdict(report):
    rd = report["ctgan_readiness"]
    assert rd["verdict"] in {"ready", "ready_with_constraints", "prepare_first"}
    # Identifier columns must always be on the drop list for CTGAN preprocessing.
    assert "transaction_id" in rd["preprocessing_drop_columns"]
    assert "user_id" in rd["preprocessing_drop_columns"]


def test_evaluation_is_deterministic(eval_df):
    r1 = evaluate_dataset(eval_df, random_state=42)
    r2 = evaluate_dataset(eval_df, random_state=42)
    assert r1["benchmark"]["detectors"] == r2["benchmark"]["detectors"]
    assert r1["sequential"]["recall_lift_from_sequence"] == r2["sequential"]["recall_lift_from_sequence"]


def test_format_report_renders(report):
    text = format_report(report)
    assert "FraudLab Dataset Evaluation Report" in text
    assert "DETECTOR BENCHMARK" in text
    assert "CTGAN READINESS" in text
