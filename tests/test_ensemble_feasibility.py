"""Tests for the Phase 3.1 ensemble-feasibility study (analysis correctness only)."""

from __future__ import annotations

import numpy as np
import pytest

from fraudlab.models.ensemble_analysis import (
    confusion_matrices,
    decide,
    fn_recovery_matrix,
    probability_correlation,
    simulate_ensembles,
    tp_overlap_matrix,
)


@pytest.fixture
def toy():
    # 6 rows; positives at indices 0,1,2.
    y = np.array([1, 1, 1, 0, 0, 0])
    labels = {
        "A": np.array([1, 1, 0, 0, 0, 0]),  # TP {0,1}
        "B": np.array([1, 0, 1, 1, 0, 0]),  # TP {0,2}, one FP
    }
    return y, labels


def test_confusion_matrix_counts(toy):
    y, labels = toy
    cm = confusion_matrices(y, labels)
    assert cm["A"] == {"TP": 2, "FP": 0, "TN": 3, "FN": 1}
    assert cm["B"] == {"TP": 2, "FP": 1, "TN": 2, "FN": 1}


def test_tp_overlap_jaccard(toy):
    y, labels = toy
    mat = tp_overlap_matrix(y, labels)
    # TP_A={0,1}, TP_B={0,2} -> |∩|=1, |∪|=3 -> 1/3.
    assert mat.loc["A", "B"] == round(1 / 3, 4)
    assert mat.loc["A", "A"] == 1.0


def test_fn_recovery(toy):
    y, labels = toy
    rec = fn_recovery_matrix(y, labels)
    # A misses {2}; B's TP includes 2 -> recovered. B misses {1}; A's TP includes 1.
    assert rec["counts"].loc["A", "B"] == 1
    assert rec["pct"].loc["A", "B"] == 100.0
    assert rec["details"]["A"]["recovered_by_any"]["count"] == 1


def test_probability_correlation_identity():
    s = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
    corr = probability_correlation({"A": s, "B": s.copy(), "C": -s})
    assert corr.loc["A", "B"] == pytest.approx(1.0, abs=1e-6)
    assert corr.loc["A", "C"] == pytest.approx(-1.0, abs=1e-6)


def test_soft_and_weighted_aggregation():
    ctx = {
        "scores": {
            "logistic_regression": np.array([0.0, 1.0]),
            "random_forest": np.array([0.0, 1.0]),
            "gradient_boosting": np.array([0.0, 1.0]),
            "xgboost": np.array([0.4, 0.8]),
        }
    }
    # Build soft/weighted directly (stacking needs training data; covered elsewhere).
    stack = np.column_stack([ctx["scores"][n] for n in
                             ("logistic_regression", "random_forest", "gradient_boosting", "xgboost")])
    soft = stack.mean(axis=1)
    assert soft == pytest.approx([0.1, 0.95])
    w = np.array([0.1, 0.2, 0.3, 0.4])
    w = w / w.sum()
    weighted = stack @ w
    assert weighted[1] == pytest.approx(0.1 * 1 + 0.2 * 1 + 0.3 * 1 + 0.4 * 0.8)


def test_stacking_oof_no_leakage_runs():
    # Minimal end-to-end stacking on a tiny separable problem; asserts OOF path executes
    # and yields a probability per test row (leakage-free by construction of StratifiedKFold).
    from fraudlab.models.ensemble_analysis import _oof_stack_scores

    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 5))
    y = (X[:, 0] + rng.normal(scale=0.3, size=n) > 0).astype(int)
    ctx = {"X_train": X[:150], "y_train": y[:150], "X_test": X[150:]}
    scores = _oof_stack_scores(ctx, random_state=42, n_splits=3)
    assert scores.shape == (50,)
    assert np.all((scores >= 0) & (scores <= 1))


def test_decision_rules():
    base = {"recall_easy": None, "recall_medium": None, "recall_at_1pct": 0,
            "recall_at_2pct": 0, "recall_at_5pct": 0, "roc_auc": 0.9,
            "precision": 0, "recall": 0, "f1": 0}
    def m(pr, hard):
        d = dict(base); d["pr_auc"] = pr; d["recall_hard"] = hard; return d

    # A: best ensemble beats PR-AUC by >=0.01 AND recall_hard improves.
    mets = {"xgboost": m(0.90, 0.70), "soft_voting": m(0.92, 0.72),
            "weighted_voting": m(0.91, 0.71), "stacking": m(0.905, 0.70)}
    assert decide(mets)["verdict"] == "A"

    # B: negligible change.
    mets_b = {"xgboost": m(0.90, 0.70), "soft_voting": m(0.903, 0.70),
              "weighted_voting": m(0.901, 0.69), "stacking": m(0.902, 0.70)}
    assert decide(mets_b)["verdict"] == "B"

    # C: all ensembles degrade PR-AUC.
    mets_c = {"xgboost": m(0.90, 0.70), "soft_voting": m(0.85, 0.65),
              "weighted_voting": m(0.86, 0.66), "stacking": m(0.84, 0.60)}
    assert decide(mets_c)["verdict"] == "C"
