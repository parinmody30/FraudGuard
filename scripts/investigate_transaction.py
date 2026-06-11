#!/usr/bin/env python
"""Run the Investigation Agent over champion+SHAP explanations (Phase 6).

Loads explanation objects (from a Phase 5 JSON file, or freshly generated via the champion
explainer), produces grounded investigation narratives (Groq if available, deterministic
fallback otherwise), prints them, and exports a JSON report.

Examples:
    py scripts/investigate_transaction.py --transaction-id T0000020057
    py scripts/investigate_transaction.py --input reports/shap_explanations.json
    py scripts/investigate_transaction.py --no-groq --transaction-id T0000020057
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

from fraudlab.agent.investigator import InvestigationAgent  # noqa: E402


def _explanations_from_args(args) -> list[dict]:
    if args.input:
        with open(args.input, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        explanations = data if isinstance(data, list) else [data]
    else:
        # Generate explanations on the fly via the champion explainer (read-only use).
        from fraudlab.config.config import BusinessType, FraudLabConfig, RiskAppetite
        from fraudlab.exporters.export import transactions_to_dataframe
        from fraudlab.explainability.shap_explainer import ChampionExplainer
        from fraudlab.pipeline import build_dataset

        config = FraudLabConfig(
            num_users=args.num_users, transactions_per_user=args.transactions_per_user,
            fraud_rate=args.fraud_rate, business_type=BusinessType(args.business_type),
            risk_appetite=RiskAppetite(args.risk_appetite), random_seed=args.seed,
        )
        tx, _ = build_dataset(config)
        df = transactions_to_dataframe(tx)
        if args.transaction_id:
            df = df[df["transaction_id"].astype(str) == args.transaction_id]
            if df.empty:
                raise SystemExit(f"transaction_id {args.transaction_id} not found.")
        else:
            df = df.head(args.limit)
        explanations = ChampionExplainer.load(args.model).explain_batch(df)

    if args.transaction_id:
        explanations = [e for e in explanations if str(e.get("transaction_id")) == args.transaction_id]
        if not explanations:
            raise SystemExit(f"transaction_id {args.transaction_id} not in explanations.")
    return explanations[: args.limit]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Investigate transactions with the AI agent.")
    p.add_argument("--input", help="Phase 5 explanations JSON (list or single object).")
    p.add_argument("--transaction-id", help="Filter / explain a single transaction id.")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--no-groq", action="store_true", help="Force deterministic offline mode.")
    p.add_argument("--num-users", type=int, default=400)
    p.add_argument("--transactions-per-user", type=int, default=50)
    p.add_argument("--fraud-rate", type=float, default=0.06)
    p.add_argument("--business-type", default="bank")
    p.add_argument("--risk-appetite", default="medium")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", default=os.path.join("models", "champion_model.joblib"))
    p.add_argument("--out", default=os.path.join("reports", "investigations.json"))
    args = p.parse_args(argv)

    explanations = _explanations_from_args(args)
    agent = InvestigationAgent.load(use_groq=not args.no_groq)
    reports = agent.investigate_batch(explanations)

    print(f"mode: {'GROQ' if agent.groq_enabled else 'FALLBACK (offline)'}  "
          f"({len(reports)} investigation(s))\n")
    for r in reports:
        print("=" * 64)
        print(f"transaction_id : {r['transaction_id']}")
        print(f"summary        : {r['summary']}")
        print("signals_detected:")
        for s in r["signals_detected"]:
            print(f"  - {s}")
        print(f"recommendation : {r['recommendation']}")
        print(f"confidence     : {r['confidence_explanation']}")
        print("next_steps:")
        for s in r["next_steps"]:
            print(f"  - {s}")
        print()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(reports, fh, indent=2, default=str)
    print(f"exported {len(reports)} investigation(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
