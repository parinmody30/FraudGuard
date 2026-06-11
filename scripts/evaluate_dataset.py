#!/usr/bin/env python
"""CLI: evaluate a FraudLab dataset (Phase 1.1).

Loads a dataset (from a file, or generated on the fly from a config/flags), runs the full
evaluation suite, prints the report, and optionally writes the report JSON.

Examples
--------
    # Evaluate an existing exported dataset
    python scripts/evaluate_dataset.py --input out/bank_seed42.csv

    # Generate a fresh dataset and evaluate it in one step
    python scripts/evaluate_dataset.py --generate --num-users 800 \
        --transactions-per-user 120 --fraud-rate 0.02 --report-json out/eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from fraudlab.config.config import BusinessType, FraudLabConfig, RiskAppetite  # noqa: E402
from fraudlab.evaluation import evaluate_dataset, format_report  # noqa: E402
from fraudlab.exporters.export import transactions_to_dataframe  # noqa: E402
from fraudlab.pipeline import build_dataset  # noqa: E402


def _load_dataframe(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path, parse_dates=["timestamp"])
    if ext == ".json":
        return pd.read_json(path, convert_dates=["timestamp"])
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input extension: {ext}")


def _generate_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    raw: dict[str, object] = {}
    if args.config:
        import yaml

        with open(args.config, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    if args.num_users is not None:
        raw["num_users"] = args.num_users
    if args.transactions_per_user is not None:
        raw["transactions_per_user"] = args.transactions_per_user
    if args.fraud_rate is not None:
        raw["fraud_rate"] = args.fraud_rate
    if args.business_type is not None:
        raw["business_type"] = args.business_type
    if args.risk_appetite is not None:
        raw["risk_appetite"] = args.risk_appetite
    if args.seed is not None:
        raw["random_seed"] = args.seed

    config = FraudLabConfig.from_dict(raw)
    transactions, _summary = build_dataset(config)
    return transactions_to_dataframe(transactions)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a FraudLab dataset.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="Path to an existing dataset (csv/json/parquet).")
    src.add_argument("--generate", action="store_true", help="Generate a dataset to evaluate.")

    parser.add_argument("--config", help="YAML config (with --generate).")
    parser.add_argument("--num-users", type=int)
    parser.add_argument("--transactions-per-user", type=int)
    parser.add_argument("--fraud-rate", type=float)
    parser.add_argument("--business-type", choices=[b.value for b in BusinessType])
    parser.add_argument("--risk-appetite", choices=[r.value for r in RiskAppetite])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--report-json", help="Optional path to write the report as JSON.")
    args = parser.parse_args(argv)

    df = _generate_dataframe(args) if args.generate else _load_dataframe(args.input)

    report = evaluate_dataset(df, test_size=args.test_size, random_state=args.random_state)
    print(format_report(report))

    if args.report_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_json)), exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\nreport JSON written to: {args.report_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
