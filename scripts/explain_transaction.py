#!/usr/bin/env python
"""Explain champion-model predictions with SHAP (Phase 5).

Loads the persisted champion bundle, scores + explains transaction(s), prints the
analyst-friendly explanation, and exports a JSON report. Transactions come from a file, a
specific --transaction-id within a generated dataset, or a freshly generated sample.

Examples:
    py scripts/explain_transaction.py --transaction-id T0000001234
    py scripts/explain_transaction.py --input out/scored_demo.csv --limit 5
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

import pandas as pd  # noqa: E402

from fraudlab.config.config import BusinessType, FraudLabConfig, RiskAppetite  # noqa: E402
from fraudlab.exporters.export import transactions_to_dataframe  # noqa: E402
from fraudlab.explainability.formatter import format_analyst_summary  # noqa: E402
from fraudlab.explainability.shap_explainer import ChampionExplainer  # noqa: E402
from fraudlab.models.persistence import DEFAULT_MODEL_PATH  # noqa: E402
from fraudlab.pipeline import build_dataset  # noqa: E402


def _load_transactions(args) -> pd.DataFrame:
    if args.input:
        ext = os.path.splitext(args.input)[1].lower()
        if ext == ".csv":
            return pd.read_csv(args.input, parse_dates=["timestamp"])
        if ext == ".json":
            return pd.read_json(args.input, convert_dates=["timestamp"])
        if ext == ".parquet":
            return pd.read_parquet(args.input)
        raise ValueError(f"Unsupported input extension: {ext}")
    config = FraudLabConfig(
        num_users=args.num_users,
        transactions_per_user=args.transactions_per_user,
        fraud_rate=args.fraud_rate,
        business_type=BusinessType(args.business_type),
        risk_appetite=RiskAppetite(args.risk_appetite),
        random_seed=args.seed,
    )
    transactions, _ = build_dataset(config)
    return transactions_to_dataframe(transactions)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Explain transactions with SHAP.")
    p.add_argument("--input", help="Transactions file (csv/json/parquet).")
    p.add_argument("--transaction-id", help="Explain a single transaction id.")
    p.add_argument("--limit", type=int, default=5, help="Max transactions to explain.")
    p.add_argument("--num-users", type=int, default=80)
    p.add_argument("--transactions-per-user", type=int, default=40)
    p.add_argument("--fraud-rate", type=float, default=0.06)
    p.add_argument("--business-type", default="bank", choices=[b.value for b in BusinessType])
    p.add_argument("--risk-appetite", default="medium", choices=[r.value for r in RiskAppetite])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default=DEFAULT_MODEL_PATH)
    p.add_argument("--out", default=os.path.join("reports", "shap_explanations.json"))
    args = p.parse_args(argv)

    if not os.path.exists(args.model):
        raise SystemExit(
            f"Champion model not found at {args.model}. Train it first:\n"
            "  python scripts/train_champion_model.py"
        )

    df = _load_transactions(args)
    if args.transaction_id:
        df = df[df["transaction_id"].astype(str) == args.transaction_id]
        if df.empty:
            raise SystemExit(f"transaction_id {args.transaction_id} not found in dataset.")
    else:
        df = df.head(args.limit)

    explainer = ChampionExplainer.load(args.model)
    explanations = explainer.explain_batch(df)

    for exp in explanations:
        print("=" * 64)
        print(json.dumps({k: v for k, v in exp.items() if k != "contributors"}, indent=2))
        print("contributors:")
        for c in exp["contributors"]:
            print(f"  {c['direction']:8s} {c['impact']:+.4f}  {c['feature']}")
        print()
        print(format_analyst_summary(exp))
        print()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(explanations, fh, indent=2, default=str)
    print(f"exported {len(explanations)} explanation(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
