"""Phase 3.1 - Model error-diversity & ensemble-feasibility study (evaluation only).

Reuses the Phase 3 preprocessing and trainer pipelines unchanged. It trains the four
in-scope models (LR / RF / GB / XGB) on the exact Phase 3 user-grouped split, then asks
whether they make *different* mistakes (TP overlap, FN recovery, tier residuals, probability
correlation) and whether fusing them (soft / weighted voting, OOF stacking) beats XGBoost.

No production ensemble is built; nothing here mutates the simulator, CTGAN, or Phase 3 code.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from fraudlab.models.metrics import compute_metrics
from fraudlab.models.preprocessing import build_design, grouped_split, make_preprocessor
from fraudlab.models.trainer import build_models, run_model

ENSEMBLE_MODELS: tuple[str, ...] = (
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "xgboost",
)
TIERS: tuple[str, ...] = ("easy", "medium", "hard")


# --------------------------------------------------------------------------- fitting

def fit_base_models(
    df: pd.DataFrame, *, test_size: float = 0.2, random_state: int = 42
) -> dict[str, Any]:
    """Train the four base models on the Phase 3 split; return test scores/labels + context."""
    design = build_design(df)
    train_idx, test_idx = grouped_split(design, test_size=test_size, random_state=random_state)

    pre = make_preprocessor(design)
    pre.fit(design.X.iloc[train_idx])  # fit on train only
    X_train = pre.transform(design.X.iloc[train_idx])
    X_test = pre.transform(design.X.iloc[test_idx])
    y_train, y_test = design.y[train_idx], design.y[test_idx]
    tiers_test = design.tiers[test_idx]

    models, _skipped = build_models(random_state=random_state)
    scores: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for name in ENSEMBLE_MODELS:
        if name not in models:
            raise RuntimeError(f"required model unavailable: {name}")
        s, l = run_model(models[name], X_train, y_train, X_test)
        scores[name], labels[name] = np.asarray(s, float), np.asarray(l, int)

    return {
        "X_train": np.asarray(X_train, float),
        "X_test": np.asarray(X_test, float),
        "y_train": y_train,
        "y_test": y_test,
        "tiers_test": tiers_test,
        "scores": scores,
        "labels": labels,
        "users_disjoint": bool(
            set(np.unique(design.groups[train_idx])).isdisjoint(
                set(np.unique(design.groups[test_idx]))
            )
        ),
        "n_test": int(len(test_idx)),
        "n_test_fraud": int(y_test.sum()),
    }


# ----------------------------------------------------------------- error-diversity

def confusion_matrices(y: np.ndarray, labels: dict[str, np.ndarray]) -> dict[str, dict[str, int]]:
    """Per-model TP / FP / TN / FN at each model's Phase 3 operating threshold."""
    y = np.asarray(y).astype(bool)
    out: dict[str, dict[str, int]] = {}
    for name, pred in labels.items():
        p = np.asarray(pred).astype(bool)
        out[name] = {
            "TP": int((p & y).sum()),
            "FP": int((p & ~y).sum()),
            "TN": int((~p & ~y).sum()),
            "FN": int((~p & y).sum()),
        }
    return out


def _tp_set(y: np.ndarray, pred: np.ndarray) -> set[int]:
    y = np.asarray(y).astype(bool)
    p = np.asarray(pred).astype(bool)
    return set(np.flatnonzero(p & y).tolist())


def tp_overlap_matrix(y: np.ndarray, labels: dict[str, np.ndarray]) -> pd.DataFrame:
    """Symmetric Jaccard TP-overlap matrix: |TP_A ∩ TP_B| / |TP_A ∪ TP_B|."""
    names = list(labels)
    tps = {n: _tp_set(y, labels[n]) for n in names}
    mat = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for a, b in combinations(names, 2):
        union = tps[a] | tps[b]
        jac = len(tps[a] & tps[b]) / len(union) if union else 0.0
        mat.loc[a, b] = mat.loc[b, a] = round(jac, 4)
    return mat


def fn_recovery_matrix(y: np.ndarray, labels: dict[str, np.ndarray]) -> dict[str, Any]:
    """For each model's missed frauds, how many each *other* model recovers (count + %)."""
    names = list(labels)
    y = np.asarray(y).astype(bool)
    fn_sets = {n: set(np.flatnonzero(~np.asarray(labels[n]).astype(bool) & y).tolist()) for n in names}
    tp_sets = {n: _tp_set(y, labels[n]) for n in names}

    counts = pd.DataFrame(0, index=names, columns=names, dtype=int)
    pct = pd.DataFrame(0.0, index=names, columns=names, dtype=float)
    details: dict[str, Any] = {}
    for misser in names:
        missed = fn_sets[misser]
        details[misser] = {"n_missed": len(missed), "recovered_by": {}}
        for other in names:
            if other == misser:
                continue
            rec = len(missed & tp_sets[other])
            counts.loc[misser, other] = rec
            pct.loc[misser, other] = round(100.0 * rec / len(missed), 2) if missed else 0.0
            details[misser]["recovered_by"][other] = {
                "count": rec,
                "pct": pct.loc[misser, other],
            }
        union_rec = set().union(*(tp_sets[o] for o in names if o != misser)) if len(names) > 1 else set()
        details[misser]["recovered_by_any"] = {
            "count": len(missed & union_rec),
            "pct": round(100.0 * len(missed & union_rec) / len(missed), 2) if missed else 0.0,
        }
    return {"counts": counts, "pct": pct, "details": details}


def tier_residuals(
    y: np.ndarray, tiers: np.ndarray, labels: dict[str, np.ndarray]
) -> dict[str, dict[str, dict[str, int]]]:
    """Per-model TP / FN split by fraud tier (easy/medium/hard)."""
    y = np.asarray(y).astype(bool)
    out: dict[str, dict[str, dict[str, int]]] = {}
    for name, pred in labels.items():
        p = np.asarray(pred).astype(bool)
        per_tier: dict[str, dict[str, int]] = {}
        for tier in TIERS:
            mask = (tiers == tier) & y
            per_tier[tier] = {
                "TP": int((p & mask).sum()),
                "FN": int((~p & mask).sum()),
            }
        out[name] = per_tier
    return out


def probability_correlation(scores: dict[str, np.ndarray]) -> pd.DataFrame:
    """Pairwise Pearson correlation of predicted fraud probabilities on the test set."""
    df = pd.DataFrame({n: np.asarray(s, float) for n, s in scores.items()})
    return df.corr(method="pearson").round(4)


# ------------------------------------------------------------------- ensembles

def _oof_stack_scores(
    ctx: dict[str, Any], *, random_state: int = 42, n_splits: int = 5
) -> np.ndarray:
    """Leakage-free stacking: level-0 OOF preds on train -> meta-LR -> test scores.

    Each base model is refit on out-of-fold partitions of the training data to produce
    unbiased level-1 features; the meta learner never sees a base model's prediction on
    rows that model was trained on.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    X_tr, y_tr, X_te = ctx["X_train"], ctx["y_train"], ctx["X_test"]
    models, _ = build_models(random_state=random_state)
    base = {n: models[n] for n in ENSEMBLE_MODELS}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros((len(y_tr), len(base)), dtype=float)
    test_accum = np.zeros((len(X_te), len(base)), dtype=float)

    for j, (name, model) in enumerate(base.items()):
        for tr, va in skf.split(X_tr, y_tr):
            fold_model = build_models(random_state=random_state)[0][name]
            fold_model.fit(X_tr[tr], y_tr[tr])
            oof[va, j] = fold_model.score(X_tr[va])
            test_accum[:, j] += fold_model.score(X_te)
        test_accum[:, j] /= n_splits

    meta = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    meta.fit(oof, y_tr)
    idx = list(meta.classes_).index(1) if 1 in meta.classes_ else 1
    return meta.predict_proba(test_accum)[:, idx]


def simulate_ensembles(
    ctx: dict[str, Any], *, pr_auc_weights: dict[str, float], random_state: int = 42
) -> dict[str, np.ndarray]:
    """Return test-set probability vectors for soft / weighted voting and OOF stacking."""
    scores = ctx["scores"]
    stack = np.column_stack([scores[n] for n in ENSEMBLE_MODELS])

    soft = stack.mean(axis=1)

    w = np.array([max(0.0, pr_auc_weights.get(n, 0.0)) for n in ENSEMBLE_MODELS], dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.full(len(ENSEMBLE_MODELS), 1 / len(ENSEMBLE_MODELS))
    weighted = stack @ w

    stacking = _oof_stack_scores(ctx, random_state=random_state)
    return {"soft_voting": soft, "weighted_voting": weighted, "stacking": stacking}


def _labels_at_topk(scores: np.ndarray, k: int) -> np.ndarray:
    n = len(scores)
    k = max(0, min(k, n))
    order = np.lexsort((np.arange(n), -np.asarray(scores, float)))
    lab = np.zeros(n, dtype=int)
    lab[order[:k]] = 1
    return lab


def ensemble_metrics(ctx: dict[str, Any], ens_scores: dict[str, np.ndarray]) -> dict[str, dict]:
    """Full metric set for XGBoost (reference) and each ensemble.

    Ensemble probabilities have no native 0.5 operating point comparable across strategies,
    so their P/R/F1 use the prevalence threshold (flag top-K where K = #test frauds); XGBoost
    keeps its Phase 3 native-threshold labels.
    """
    y, tiers = ctx["y_test"], ctx["tiers_test"]
    k = int(np.asarray(y).sum())

    out: dict[str, dict] = {}
    out["xgboost"] = compute_metrics(y, ctx["scores"]["xgboost"], ctx["labels"]["xgboost"], tiers)
    for name, sc in ens_scores.items():
        out[name] = compute_metrics(y, sc, _labels_at_topk(sc, k), tiers)
    return out


# --------------------------------------------------------------------- decision

def decide(metrics: dict[str, dict], *, pr_auc_min_gain: float = 0.01) -> dict[str, Any]:
    """Apply the Phase 3.1 decision rules vs the XGBoost champion."""
    ref = metrics["xgboost"]
    ref_pr, ref_hard = ref["pr_auc"], ref["recall_hard"]

    rows = []
    for name in ("soft_voting", "weighted_voting", "stacking"):
        m = metrics[name]
        rows.append(
            {
                "ensemble": name,
                "pr_auc": m["pr_auc"],
                "delta_pr_auc": round(m["pr_auc"] - ref_pr, 4),
                "recall_hard": m["recall_hard"],
                "delta_recall_hard": round((m["recall_hard"] or 0) - (ref_hard or 0), 4),
            }
        )
    best = max(rows, key=lambda r: r["pr_auc"])

    improves_pr = best["delta_pr_auc"] >= pr_auc_min_gain
    improves_hard = best["delta_recall_hard"] > 0
    any_degrade_all = all(r["delta_pr_auc"] < 0 for r in rows)

    if improves_pr and improves_hard:
        verdict, rec = "A", "PROCEED WITH ENSEMBLE"
    elif any_degrade_all or best["delta_pr_auc"] < -pr_auc_min_gain:
        verdict, rec = "C", "ABANDON ENSEMBLES; XGBOOST REMAINS CHAMPION"
    else:
        verdict, rec = "B", "USE XGBOOST ONLY; AVOID ENSEMBLE COMPLEXITY"

    return {
        "verdict": verdict,
        "recommendation": rec,
        "pr_auc_min_gain": pr_auc_min_gain,
        "reference_xgboost": {"pr_auc": ref_pr, "recall_hard": ref_hard},
        "ensembles": rows,
        "best_ensemble": best["ensemble"],
    }


def run_feasibility_study(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    pr_auc_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Full study: fit bases, run all diversity analyses, simulate ensembles, decide."""
    ctx = fit_base_models(df, test_size=test_size, random_state=random_state)

    # Default voting weights proportional to each model's own test PR-AUC.
    base_metrics = {
        n: compute_metrics(ctx["y_test"], ctx["scores"][n], ctx["labels"][n], ctx["tiers_test"])
        for n in ENSEMBLE_MODELS
    }
    if pr_auc_weights is None:
        pr_auc_weights = {n: max(0.0, base_metrics[n]["pr_auc"]) for n in ENSEMBLE_MODELS}

    ens_scores = simulate_ensembles(ctx, pr_auc_weights=pr_auc_weights, random_state=random_state)
    ens_metrics = ensemble_metrics(ctx, ens_scores)
    decision = decide(ens_metrics)

    return {
        "context": {
            "n_test": ctx["n_test"],
            "n_test_fraud": ctx["n_test_fraud"],
            "users_disjoint": ctx["users_disjoint"],
        },
        "base_metrics": base_metrics,
        "confusion": confusion_matrices(ctx["y_test"], ctx["labels"]),
        "tp_overlap": tp_overlap_matrix(ctx["y_test"], ctx["labels"]),
        "fn_recovery": fn_recovery_matrix(ctx["y_test"], ctx["labels"]),
        "tier_residuals": tier_residuals(ctx["y_test"], ctx["tiers_test"], ctx["labels"]),
        "probability_correlation": probability_correlation(ctx["scores"]),
        "ensemble_metrics": ens_metrics,
        "voting_weights": {n: round(w, 4) for n, w in pr_auc_weights.items()},
        "decision": decision,
    }
