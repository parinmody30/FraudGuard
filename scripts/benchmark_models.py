#!/usr/bin/env python
"""Phase 3 benchmark: Simulator vs Hybrid fraud-detection leaderboard.

Builds two datasets that differ *only* in generation method:
  A) Simulator-only      - ``build_dataset``.
  B) Hybrid              - ``build_hybrid_dataset`` (simulator hard fraud + CTGAN soft rows),
     matched to A on seed, row count, fraud rate, and fraud mix.

It trains every available detector on each dataset, compares them, and writes:
    reports/benchmark_results.json
    reports/benchmark_results.md
    reports/leaderboard.csv

Requires a trained CTGAN at ``models/ctgan.pkl`` (run ``scripts/train_ctgan.py`` first).

Example
-------
    python scripts/benchmark_models.py --num-users 400 --transactions-per-user 50 \
        --fraud-rate 0.06 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Print UTF-8 regardless of the console code page (Windows defaults to cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - older interpreters / redirected streams
    pass

import pandas as pd  # noqa: E402

from fraudlab.config.config import BusinessType, FraudLabConfig, RiskAppetite  # noqa: E402
from fraudlab.exporters.export import transactions_to_dataframe  # noqa: E402
from fraudlab.gan.merger import build_hybrid_dataset  # noqa: E402
from fraudlab.gan.sampler import CTGANSampler  # noqa: E402
from fraudlab.gan.trainer import DEFAULT_MODEL_PATH, load_artifacts  # noqa: E402
from fraudlab.models.benchmark import run_full_benchmark  # noqa: E402
from fraudlab.models.leaderboard import LEADERBOARD_METRICS  # noqa: E402
from fraudlab.pipeline import build_dataset  # noqa: E402

REPORTS_DIR = "reports"


def _generate_datasets(args) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    config = FraudLabConfig(
        num_users=args.num_users,
        transactions_per_user=args.transactions_per_user,
        fraud_rate=args.fraud_rate,
        business_type=BusinessType(args.business_type),
        risk_appetite=RiskAppetite(args.risk_appetite),
        random_seed=args.seed,
    )
    transactions, _ = build_dataset(config)
    simulator_df = transactions_to_dataframe(transactions)

    if not os.path.exists(args.model):
        raise SystemExit(
            f"CTGAN model not found at {args.model}. Train it first:\n"
            "  python scripts/train_ctgan.py --generate --num-users 250 "
            "--transactions-per-user 50 --fraud-rate 0.08 --epochs 120 --seed 42"
        )
    model, metadata, _ = load_artifacts(model_path=args.model, metadata_path=args.metadata)
    sampler = CTGANSampler(model, metadata, seed=args.seed)

    realized_rate = float(simulator_df["is_fraud"].mean())
    hybrid_df, composition = build_hybrid_dataset(
        simulator_df,
        sampler,
        n_rows=len(simulator_df),
        fraud_rate=realized_rate,
        fraud_mix=config.fraud_mix,
        seed=args.seed,
    )
    meta = {
        "config": {
            "num_users": config.num_users,
            "transactions_per_user": config.transactions_per_user,
            "fraud_rate": config.fraud_rate,
            "business_type": config.business_type.value,
            "risk_appetite": config.risk_appetite.value,
            "seed": config.random_seed,
        },
        "matched_fraud_rate": round(realized_rate, 4),
        "hybrid_composition": composition,
    }
    return simulator_df, hybrid_df, meta


def _stacked_leaderboard(comparison) -> pd.DataFrame:
    sim = comparison["simulator_leaderboard"].copy()
    hyb = comparison["hybrid_leaderboard"].copy()
    sim.insert(0, "dataset", "simulator")
    hyb.insert(0, "dataset", "hybrid")
    return pd.concat([sim, hyb], ignore_index=True)


def _to_records(report: dict, meta: dict) -> dict:
    comp = report["comparison"]
    return {
        "datasets": meta,
        "simulator": {
            "split": report["simulator"]["split"],
            "skipped_libraries": report["simulator"]["skipped_libraries"],
            "models": report["simulator"]["models"],
        },
        "hybrid": {
            "split": report["hybrid"]["split"],
            "skipped_libraries": report["hybrid"]["skipped_libraries"],
            "models": report["hybrid"]["models"],
        },
        "simulator_leaderboard": comp["simulator_leaderboard"].to_dict(orient="records"),
        "hybrid_leaderboard": comp["hybrid_leaderboard"].to_dict(orient="records"),
        "improvement": comp["improvement"].to_dict(orient="records"),
        "sequential_findings": comp["sequential_findings"],
        "decision": comp["decision"],
    }


def _markdown(report: dict, meta: dict) -> str:
    comp = report["comparison"]
    L: list[str] = []
    a = L.append
    a("# FraudLab Phase 3 — Fraud Detection Benchmark\n")

    a("## Datasets\n")
    sp_s = report["simulator"]["split"]
    sp_h = report["hybrid"]["split"]
    a(f"- Config: {meta['config']}")
    a(f"- Matched fraud rate: **{meta['matched_fraud_rate']}**")
    a(f"- Simulator: {sp_s['n_rows']} rows, {sp_s['n_users']} users, fraud {sp_s['fraud_rate']}, "
      f"test fraud {sp_s['test_fraud']}, users_disjoint={sp_s['users_disjoint']}")
    a(f"- Hybrid: {sp_h['n_rows']} rows, {sp_h['n_users']} users, fraud {sp_h['fraud_rate']}, "
      f"test fraud {sp_h['test_fraud']}, users_disjoint={sp_h['users_disjoint']}")
    a(f"- Skipped libraries: {report['simulator']['skipped_libraries']}\n")

    def _lb_md(df: pd.DataFrame) -> None:
        cols = ["rank", "model", *LEADERBOARD_METRICS]
        cols = [c for c in cols if c in df.columns]
        a("| " + " | ".join(cols) + " |")
        a("|" + "|".join(["---"] * len(cols)) + "|")
        for _, row in df.iterrows():
            a("| " + " | ".join(str(row[c]) for c in cols) + " |")
        a("")

    a("## Simulator leaderboard (by PR-AUC)\n")
    _lb_md(comp["simulator_leaderboard"])
    a("## Hybrid leaderboard (by PR-AUC)\n")
    _lb_md(comp["hybrid_leaderboard"])

    a("## Hybrid - Simulator improvement\n")
    imp = comp["improvement"]
    key_cols = ["model", "sim_pr_auc", "hyb_pr_auc", "delta_pr_auc",
                "sim_recall_hard", "hyb_recall_hard", "delta_recall_hard"]
    key_cols = [c for c in key_cols if c in imp.columns]
    a("| " + " | ".join(key_cols) + " |")
    a("|" + "|".join(["---"] * len(key_cols)) + "|")
    for _, row in imp.iterrows():
        a("| " + " | ".join(str(row[c]) for c in key_cols) + " |")
    a("")

    a("## Sequential (LSTM) findings — recall_hard\n")
    for ds, f in comp["sequential_findings"].items():
        if f.get("lstm_available"):
            a(f"- **{ds}**: LSTM recall_hard={f['lstm_recall_hard']} vs best row-level "
              f"({f['best_row_level_model']}) recall_hard={f['best_row_level_recall_hard']} "
              f"→ Δ={f['lstm_minus_best_row_level']} (LSTM helps: {f['lstm_helps_hard']})")
        else:
            a(f"- **{ds}**: LSTM skipped ({f.get('reason')})")
    a("")

    d = comp["decision"]
    a("## Decision\n")
    a(f"- PR-AUC deltas: {d['pr_auc_delta_stats']}")
    a(f"- recall_hard deltas: {d['recall_hard_delta_stats']}")
    a(f"\n**RECOMMENDATION ({d['verdict']}): {d['recommendation']}**")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simulator vs Hybrid detection benchmark.")
    parser.add_argument("--num-users", type=int, default=400)
    parser.add_argument("--transactions-per-user", type=int, default=50)
    parser.add_argument("--fraud-rate", type=float, default=0.06)
    parser.add_argument("--business-type", default="bank", choices=[b.value for b in BusinessType])
    parser.add_argument("--risk-appetite", default="medium", choices=[r.value for r in RiskAppetite])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata", default="models/ctgan_metadata.json")
    parser.add_argument("--reports-dir", default=REPORTS_DIR)
    args = parser.parse_args(argv)

    simulator_df, hybrid_df, meta = _generate_datasets(args)
    report = run_full_benchmark(
        simulator_df, hybrid_df, test_size=args.test_size, random_state=args.seed
    )

    os.makedirs(args.reports_dir, exist_ok=True)
    json_path = os.path.join(args.reports_dir, "benchmark_results.json")
    md_path = os.path.join(args.reports_dir, "benchmark_results.md")
    csv_path = os.path.join(args.reports_dir, "leaderboard.csv")

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(_to_records(report, meta), fh, indent=2, default=str)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(_markdown(report, meta))
    _stacked_leaderboard(report["comparison"]).to_csv(csv_path, index=False)

    print(_markdown(report, meta))
    print(f"\nwritten:\n  {json_path}\n  {md_path}\n  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
