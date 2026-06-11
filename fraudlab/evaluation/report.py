"""Top-level evaluation orchestrator and human-readable report formatting."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fraudlab.evaluation.benchmark import run_benchmark
from fraudlab.evaluation.features import build_feature_bundle
from fraudlab.evaluation.leakage import detect_leakage
from fraudlab.evaluation.metrics import dataset_metrics
from fraudlab.evaluation.readiness import assess_ctgan_readiness
from fraudlab.evaluation.sequential import analyze_sequential_complexity


def evaluate_dataset(
    df: pd.DataFrame, *, test_size: float = 0.3, random_state: int = 42
) -> dict[str, Any]:
    """Run the full evaluation suite over a transaction DataFrame.

    Returns a single nested report dict with sections: ``metrics``, ``benchmark``,
    ``leakage``, ``sequential``, and ``ctgan_readiness``. Pure analysis — the input frame
    is never mutated and no data is generated.
    """
    metrics = dataset_metrics(df)

    bundle = build_feature_bundle(df)
    benchmark = run_benchmark(bundle, test_size=test_size, random_state=random_state)

    leakage = detect_leakage(df)
    sequential = analyze_sequential_complexity(
        df, test_size=test_size, random_state=random_state
    )
    ctgan_readiness = assess_ctgan_readiness(df, sequential)

    return {
        "metrics": metrics,
        "benchmark": benchmark,
        "leakage": leakage,
        "sequential": sequential,
        "ctgan_readiness": ctgan_readiness,
    }


def _fmt(v: object) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def format_report(report: dict[str, Any]) -> str:
    """Render the report dict as a readable plain-text summary."""
    m = report["metrics"]
    b = report["benchmark"]
    lk = report["leakage"]
    sq = report["sequential"]
    rd = report["ctgan_readiness"]

    lines: list[str] = []
    add = lines.append
    bar = "=" * 70

    add(bar)
    add("FraudLab Dataset Evaluation Report (Phase 1.1)")
    add(bar)

    # --- key metrics ---
    add("\n[1] KEY METRICS")
    add(f"  rows                 : {m['n_rows']}   columns: {m['n_columns']}   users: {m['n_users']}")
    add(f"  fraud / legit        : {m['n_fraud']} / {m['n_legit']}  (rate {m['fraud_rate']:.4f}, imbalance {m['imbalance_ratio']}:1)")
    add(f"  fraud per tier       : {m['fraud_per_tier']}")
    add(f"  fraud tier share     : {m['fraud_tier_share']}")
    add(f"  scenarios present    : {m['n_fraud_scenarios_present']} / 14")
    add(f"  duplicate rows       : {m['duplicate_rows']}")
    add(f"  amount p50 legit/fraud: {m['amount_by_class']['legit_p50']:.0f} / {m['amount_by_class']['fraud_p50']:.0f}")

    # --- detector benchmark ---
    add("\n[2] DETECTOR BENCHMARK")
    add(f"  protocol: {b['protocol']}")
    sp = b["split"]
    add(f"  split: train {sp['train_rows']} (fraud {sp['train_fraud']}) | test {sp['test_rows']} (fraud {sp['test_fraud']}) | users disjoint: {sp['users_disjoint']}")
    add(f"  {'detector':<20}{'roc_auc':>9}{'pr_auc':>9}{'F1':>8}{'rec_easy':>10}{'rec_med':>9}{'rec_hard':>10}")
    for name, r in b["detectors"].items():
        add(
            f"  {name:<20}{_fmt(r['roc_auc']):>9}{_fmt(r['pr_auc']):>9}{_fmt(r['f1_at_prevalence']):>8}"
            f"{_fmt(r['recall_easy']):>10}{_fmt(r['recall_medium']):>9}{_fmt(r['recall_hard']):>10}"
        )

    # --- leakage ---
    add("\n[3] LEAKAGE FINDINGS")
    add(f"  clean: {lk['clean']}   (features checked: {lk['n_features_checked']}, AUC thr {lk['auc_threshold']}, MI thr {lk['mi_threshold']})")
    for c in lk["construction_leakage"]:
        add(f"  construction: '{c['column']}' leaks_label={c['leaks_label']} -> excluded ({c['reason']})")
    if lk["single_feature_leakage"]:
        for f in lk["single_feature_leakage"]:
            add(f"  LEAK: {f['feature']} auc={f['univariate_auc']} mi={f['norm_mutual_info']}")
    else:
        add("  no single usable feature separates the classes (no leakage)")
    add("  top usable features by univariate AUC:")
    for f in lk["top_features_by_auc"][:5]:
        add(f"    {f['feature']:<22} auc={f['univariate_auc']:.4f}  mi={f['norm_mutual_info']:.4f}")

    # --- sequential complexity ---
    add("\n[4] SEQUENTIAL COMPLEXITY")
    add(f"  window: {sq['window']} txns   added features: {len(sq['sequence_features'])}")
    add(f"  per-row recall          : {sq['per_row_recall']}")
    add(f"  +sequence recall        : {sq['sequence_augmented_recall']}")
    add(f"  recall lift from sequence: {sq['recall_lift_from_sequence']}")
    add(f"  mean seq-feature AUC/tier: {sq['tier_separation_on_sequence_features']}")
    add(f"  hard fraud is sequential : {sq['hard_is_sequential']}")

    # --- ctgan readiness ---
    add("\n[5] CTGAN READINESS")
    add(f"  verdict : {rd['verdict']}")
    add(f"  headline: {rd['headline']}")
    for f in rd["findings"]:
        add(f"  - [{f['status']}] {f['dimension']}: {f['detail']}")
    if rd["blockers"]:
        add("  blockers:")
        for blk in rd["blockers"]:
            add(f"    * {blk}")
    add("  recommendations:")
    for rec in rd["recommendations"]:
        add(f"    * {rec}")

    add("\n" + bar)
    return "\n".join(lines)
