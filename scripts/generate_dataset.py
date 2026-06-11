#!/usr/bin/env python
"""CLI entry point: load config, generate dataset, export, print summary.

Examples
--------
    python scripts/generate_dataset.py --config configs/sample_bank.yaml
    python scripts/generate_dataset.py --business-type upi --num-users 200 \
        --transactions-per-user 80 --fraud-rate 0.03 --format parquet --out out/upi.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Make the package importable when run as a script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fraudlab.config.config import (  # noqa: E402
    BusinessType,
    FraudLabConfig,
    OutputFormat,
    RiskAppetite,
)
from fraudlab.exporters.export import export_dataset  # noqa: E402
from fraudlab.pipeline import build_dataset  # noqa: E402


def _build_config(args: argparse.Namespace) -> FraudLabConfig:
    # Start from the raw config inputs (not a constructed config) so that CLI overrides
    # are applied *before* validation/derivation. This matters for risk_appetite, whose
    # default fraud_mix must be re-derived when the appetite is overridden.
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
    if args.noise_level is not None:
        raw["noise_level"] = args.noise_level
    if args.seed is not None:
        raw["random_seed"] = args.seed
    if args.format is not None:
        raw["output_format"] = args.format

    return FraudLabConfig.from_dict(raw)


def _default_out_path(config: FraudLabConfig) -> str:
    ext = config.output_format.value
    return os.path.join("out", f"{config.business_type.value}_seed{config.random_seed}.{ext}")


def _print_summary(summary: dict, out_path: str) -> None:
    print("\n" + "=" * 48)
    print("FraudLab Generator - generation summary")
    print("=" * 48)
    print(f"business type           : {summary['business_type']}")
    print(f"random seed             : {summary['random_seed']}")
    print(f"users                   : {summary['num_users']}")
    print(f"transactions per user   : {summary['transactions_per_user']}")
    print(f"total transactions      : {summary['total_transactions']}")
    print(f"fraud count             : {summary['fraud_count']}")
    print(f"fraud percentage        : {summary['fraud_percentage']}%")
    print(f"generation time (s)     : {summary['generation_time_seconds']}")
    print("\nfraud type distribution:")
    dist = summary["fraud_type_distribution"]
    for name in sorted(dist, key=lambda k: dist[k], reverse=True):
        print(f"  {name:<22}: {dist[name]}")
    print(f"\noutput written to       : {out_path}")
    print("=" * 48 + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic fraud dataset.")
    parser.add_argument("--config", help="Path to a YAML config file.")
    parser.add_argument("--num-users", type=int)
    parser.add_argument("--transactions-per-user", type=int)
    parser.add_argument("--fraud-rate", type=float)
    parser.add_argument("--business-type", choices=[b.value for b in BusinessType])
    parser.add_argument("--risk-appetite", choices=[r.value for r in RiskAppetite])
    parser.add_argument("--noise-level", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--format", choices=[o.value for o in OutputFormat])
    parser.add_argument("--out", help="Output file path (default: out/<business>_seed<seed>.<ext>).")
    parser.add_argument(
        "--summary-json", help="Optional path to also write the summary as JSON."
    )
    args = parser.parse_args(argv)

    config = _build_config(args)
    transactions, summary = build_dataset(config)

    out_path = args.out or _default_out_path(config)
    export_dataset(transactions, out_path, config.output_format)

    _print_summary(summary, out_path)

    if args.summary_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_json)), exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
