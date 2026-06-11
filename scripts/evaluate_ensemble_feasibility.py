#!/usr/bin/env python
"""Phase 3.1 - Ensemble feasibility study (Simulator-only, evaluation only).

Generates the Phase 3 simulator dataset (seed 42, same config), runs the error-diversity
and ensemble-simulation analyses, and writes:
    reports/ensemble_feasibility.json
    reports/ensemble_feasibility.md
    reports/model_overlap_matrix.csv

Example:
    python scripts/evaluate_ensemble_feasibility.py --num-users 400 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from fraudlab.config.config import BusinessType, FraudLabConfig, RiskAppetite  # noqa: E402
from fraudlab.exporters.export import transactions_to_dataframe  # noqa: E402
from fraudlab.models.ensemble_analysis import ENSEMBLE_MODELS, run_feasibility_study  # noqa: E402
from fraudlab.pipeline import build_dataset  # noqa: E402

METRIC_KEYS = (
    "roc_auc", "pr_auc", "precision", "recall", "f1",
    "recall_easy", "recall_medium", "recall_hard",
    "recall_at_1pct", "recall_at_2pct", "recall_at_5pct",
)


def _df_md(df) -> str:
    """Render a labelled DataFrame to a Markdown table without external deps."""
    cols = list(df.columns)
    lines = ["| | " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join(["---"] * (len(cols) + 1)) + "|"]
    for idx, row in df.iterrows():
        lines.append("| " + str(idx) + " | " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def _markdown(r: dict) -> str:
    L: list[str] = []
    a = L.append
    a("# FraudLab Phase 3.1 - Ensemble Feasibility Study\n")
    c = r["context"]
    a(f"- Test rows: {c['n_test']}, test fraud: {c['n_test_fraud']}, users_disjoint: {c['users_disjoint']}")
    a(f"- Voting weights (PR-AUC normalized): {r['decision'] and r['voting_weights']}\n")

    a("## 1. Confusion matrices (Phase 3 operating threshold)\n")
    a("| model | TP | FP | TN | FN |")
    a("|---|---|---|---|---|")
    for n, cm in r["confusion"].items():
        a(f"| {n} | {cm['TP']} | {cm['FP']} | {cm['TN']} | {cm['FN']} |")
    a("")

    a("## 2. TP overlap matrix (Jaccard)\n")
    a(_df_md(r["tp_overlap"]))
    a("")

    a("## 3. FN recovery (% of a model's missed frauds recovered by another)\n")
    a(_df_md(r["fn_recovery"]["pct"]))
    a("")
    a("Recovered by ANY other model:")
    for misser, d in r["fn_recovery"]["details"].items():
        a(f"- {misser}: missed {d['n_missed']}, recovered by any = {d['recovered_by_any']['pct']}%")
    a("")

    a("## 4. Fraud-tier residuals (TP / FN per tier)\n")
    a("| model | easy TP | easy FN | medium TP | medium FN | hard TP | hard FN |")
    a("|---|---|---|---|---|---|---|")
    for n, t in r["tier_residuals"].items():
        a(f"| {n} | {t['easy']['TP']} | {t['easy']['FN']} | {t['medium']['TP']} | "
          f"{t['medium']['FN']} | {t['hard']['TP']} | {t['hard']['FN']} |")
    a("")

    a("## 5. Predicted-probability correlation (Pearson)\n")
    a(_df_md(r["probability_correlation"]))
    a("")

    a("## 6. Ensemble benchmark\n")
    a("| model | " + " | ".join(METRIC_KEYS) + " |")
    a("|" + "|".join(["---"] * (len(METRIC_KEYS) + 1)) + "|")
    for n in ("xgboost", "soft_voting", "weighted_voting", "stacking"):
        m = r["ensemble_metrics"][n]
        a(f"| {n} | " + " | ".join(str(m.get(k)) for k in METRIC_KEYS) + " |")
    a("")

    d = r["decision"]
    a("## 7. Decision\n")
    a(f"- Reference XGBoost: PR-AUC={d['reference_xgboost']['pr_auc']}, "
      f"recall_hard={d['reference_xgboost']['recall_hard']}")
    for e in d["ensembles"]:
        a(f"- {e['ensemble']}: PR-AUC={e['pr_auc']} (Δ {e['delta_pr_auc']}), "
          f"recall_hard={e['recall_hard']} (Δ {e['delta_recall_hard']})")
    a(f"\n**RECOMMENDATION ({d['verdict']}): {d['recommendation']}**")
    return "\n".join(L)


def _serialize(r: dict) -> dict:
    out = dict(r)
    out["tp_overlap"] = r["tp_overlap"].to_dict()
    out["probability_correlation"] = r["probability_correlation"].to_dict()
    out["fn_recovery"] = {
        "counts": r["fn_recovery"]["counts"].to_dict(),
        "pct": r["fn_recovery"]["pct"].to_dict(),
        "details": r["fn_recovery"]["details"],
    }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ensemble feasibility study.")
    p.add_argument("--num-users", type=int, default=400)
    p.add_argument("--transactions-per-user", type=int, default=50)
    p.add_argument("--fraud-rate", type=float, default=0.06)
    p.add_argument("--business-type", default="bank", choices=[b.value for b in BusinessType])
    p.add_argument("--risk-appetite", default="medium", choices=[r.value for r in RiskAppetite])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--reports-dir", default="reports")
    args = p.parse_args(argv)

    config = FraudLabConfig(
        num_users=args.num_users,
        transactions_per_user=args.transactions_per_user,
        fraud_rate=args.fraud_rate,
        business_type=BusinessType(args.business_type),
        risk_appetite=RiskAppetite(args.risk_appetite),
        random_seed=args.seed,
    )
    transactions, _ = build_dataset(config)
    df = transactions_to_dataframe(transactions)

    report = run_feasibility_study(df, test_size=args.test_size, random_state=args.seed)

    os.makedirs(args.reports_dir, exist_ok=True)
    with open(os.path.join(args.reports_dir, "ensemble_feasibility.json"), "w", encoding="utf-8") as fh:
        json.dump(_serialize(report), fh, indent=2, default=str)
    with open(os.path.join(args.reports_dir, "ensemble_feasibility.md"), "w", encoding="utf-8") as fh:
        fh.write(_markdown(report))
    report["tp_overlap"].to_csv(os.path.join(args.reports_dir, "model_overlap_matrix.csv"))

    print(_markdown(report))
    print(f"\nRECOMMENDATION ({report['decision']['verdict']}): {report['decision']['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
