"""Shared helpers for the CTGAN CLI scripts (dataset loading / generation)."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from fraudlab.config.config import BusinessType, FraudLabConfig, RiskAppetite  # noqa: E402
from fraudlab.exporters.export import transactions_to_dataframe  # noqa: E402
from fraudlab.pipeline import build_dataset  # noqa: E402


def add_source_args(parser: argparse.ArgumentParser) -> None:
    """Add the mutually exclusive --input / --generate dataset-source options."""
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="Existing simulator dataset (csv/json/parquet).")
    src.add_argument("--generate", action="store_true", help="Generate a simulator dataset.")
    parser.add_argument("--config")
    parser.add_argument("--num-users", type=int)
    parser.add_argument("--transactions-per-user", type=int)
    parser.add_argument("--fraud-rate", type=float)
    parser.add_argument("--business-type", choices=[b.value for b in BusinessType])
    parser.add_argument("--risk-appetite", choices=[r.value for r in RiskAppetite])
    parser.add_argument("--sim-seed", type=int, help="Simulator random_seed when generating.")


def load_dataframe(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path, parse_dates=["timestamp"])
    if ext == ".json":
        return pd.read_json(path, convert_dates=["timestamp"])
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input extension: {ext}")


def resolve_dataframe(args: argparse.Namespace) -> tuple[pd.DataFrame, FraudLabConfig | None]:
    """Return (simulator_df, config) from --input or --generate flags."""
    if getattr(args, "input", None):
        return load_dataframe(args.input), None

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
    if args.sim_seed is not None:
        raw["random_seed"] = args.sim_seed

    config = FraudLabConfig.from_dict(raw)
    transactions, _ = build_dataset(config)
    return transactions_to_dataframe(transactions), config
