"""Benchmark orchestration: run all detectors on one dataset, and compare two datasets.

For each dataset the pipeline is: build a leakage-safe design, make a user-grouped 80/20
split, fit the preprocessor on the *train* fold only, transform both folds, train every
available detector, and score on the test fold. A per-user LSTM is added when the dataset
contains hard (sequential) fraud. The comparison step builds per-dataset leaderboards, the
Simulator-vs-Hybrid improvement table, the sequential finding, and the final decision.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fraudlab.models.leaderboard import build_leaderboard, compute_improvement, decide
from fraudlab.models.metrics import compute_metrics
from fraudlab.models.preprocessing import build_design, grouped_split, make_preprocessor
from fraudlab.models.sequential import run_lstm
from fraudlab.models.trainer import build_models, run_model

SEQUENTIAL_MODEL = "lstm"


def run_dataset_benchmark(
    df: pd.DataFrame,
    *,
    name: str,
    test_size: float = 0.2,
    random_state: int = 42,
    include_sequential: bool = True,
) -> dict[str, Any]:
    """Train and score every detector on one dataset; return per-model metrics + metadata."""
    design = build_design(df)
    train_idx, test_idx = grouped_split(design, test_size=test_size, random_state=random_state)

    preprocessor = make_preprocessor(design)
    X_train_raw = design.X.iloc[train_idx]
    X_test_raw = design.X.iloc[test_idx]
    preprocessor.fit(X_train_raw)  # fit on train only -> no leakage from test
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    y_train, y_test = design.y[train_idx], design.y[test_idx]
    tiers_test = design.tiers[test_idx]

    models, skipped = build_models(random_state=random_state)
    results: dict[str, Any] = {}
    for mname, model in models.items():
        scores, labels = run_model(model, X_train, y_train, X_test)
        results[mname] = compute_metrics(y_test, scores, labels, tiers_test)
        results[mname]["kind"] = getattr(model, "kind", "supervised")

    has_hard = bool((design.tiers == "hard").any())
    if include_sequential and has_hard:
        X_all = preprocessor.transform(design.X)
        results[SEQUENTIAL_MODEL] = run_lstm(
            design, np.asarray(X_all, dtype=np.float32), train_idx, test_idx,
            random_state=random_state,
        )

    split = {
        "n_rows": int(len(df)),
        "n_users": int(np.unique(design.groups).size),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_fraud": int(y_train.sum()),
        "test_fraud": int(y_test.sum()),
        "users_disjoint": bool(
            set(np.unique(design.groups[train_idx])).isdisjoint(
                set(np.unique(design.groups[test_idx]))
            )
        ),
        "fraud_rate": round(float(design.y.mean()), 4),
        "has_hard_fraud": has_hard,
    }
    return {
        "name": name,
        "split": split,
        "skipped_libraries": skipped,
        "models": results,
    }


def _best_row_level(models: dict[str, dict]) -> str | None:
    """Name of the highest-PR-AUC non-sequential, non-skipped model."""
    best, best_pr = None, -1.0
    for name, m in models.items():
        if name == SEQUENTIAL_MODEL or m.get("skipped"):
            continue
        pr = m.get("pr_auc")
        if isinstance(pr, (int, float)) and not np.isnan(pr) and pr > best_pr:
            best, best_pr = name, float(pr)
    return best


def _sequential_finding(name: str, models: dict[str, dict]) -> dict[str, Any]:
    """Compare LSTM vs the best row-level model on recall_hard for one dataset."""
    lstm = models.get(SEQUENTIAL_MODEL)
    best = _best_row_level(models)
    if lstm is None or lstm.get("skipped"):
        return {
            "dataset": name,
            "lstm_available": False,
            "reason": (lstm or {}).get("reason", "sequential model not run"),
            "best_row_level_model": best,
        }
    best_hard = models[best]["recall_hard"] if best else None
    lstm_hard = lstm.get("recall_hard")
    delta = (
        round(float(lstm_hard) - float(best_hard), 4)
        if isinstance(lstm_hard, (int, float)) and isinstance(best_hard, (int, float))
        else None
    )
    return {
        "dataset": name,
        "lstm_available": True,
        "best_row_level_model": best,
        "best_row_level_recall_hard": best_hard,
        "lstm_recall_hard": lstm_hard,
        "lstm_minus_best_row_level": delta,
        "lstm_helps_hard": bool(delta is not None and delta > 0),
    }


def compare_datasets(
    simulator: dict[str, Any], hybrid: dict[str, Any]
) -> dict[str, Any]:
    """Build leaderboards, improvement table, sequential findings, and the decision."""
    sim_models = simulator["models"]
    hyb_models = hybrid["models"]

    sim_lb = build_leaderboard(sim_models)
    hyb_lb = build_leaderboard(hyb_models)
    improvement = compute_improvement(sim_models, hyb_models)
    decision = decide(improvement)

    sequential = {
        "simulator": _sequential_finding("simulator", sim_models),
        "hybrid": _sequential_finding("hybrid", hyb_models),
    }

    return {
        "simulator_leaderboard": sim_lb,
        "hybrid_leaderboard": hyb_lb,
        "improvement": improvement,
        "sequential_findings": sequential,
        "decision": decision,
    }


def run_full_benchmark(
    simulator_df: pd.DataFrame,
    hybrid_df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """End-to-end: benchmark both datasets and compare them."""
    simulator = run_dataset_benchmark(
        simulator_df, name="simulator", test_size=test_size, random_state=random_state
    )
    hybrid = run_dataset_benchmark(
        hybrid_df, name="hybrid", test_size=test_size, random_state=random_state
    )
    comparison = compare_datasets(simulator, hybrid)
    return {"simulator": simulator, "hybrid": hybrid, "comparison": comparison}
