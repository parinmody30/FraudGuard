#!/usr/bin/env python
"""Score transactions with the persisted champion model (Phase 4).

Loads the champion bundle, scores an input dataset (or a freshly generated simulator
dataset), and writes a scored dataset with ``fraud_probability``, ``decision``, and
``confidence`` columns.

Examples:
    python scripts/score_transactions.py --input out/bank_seed42.csv --out out/scored.csv
    python scripts/score_transactions.py --generate --num-users 50 --out out/scored.csv
"""

from __future__ import annotations

import argparse
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
from fraudlab.models.persistence import DEFAULT_MODEL_PATH  # noqa: E402
from fraudlab.models.predictor import ChampionPredictor  # noqa: E402
from fraudlab.pipeline import build_dataset  # noqa: E402


def _load_input(args) -> pd.DataFrame:
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


def _write(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext == ".json":
        df.to_json(path, orient="records", date_format="iso", indent=2)
    elif ext == ".parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output extension: {ext}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Score transactions with the champion model.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="Transactions to score (csv/json/parquet).")
    src.add_argument("--generate", action="store_true", help="Generate a dataset to score.")
    p.add_argument("--num-users", type=int, default=50)
    p.add_argument("--transactions-per-user", type=int, default=50)
    p.add_argument("--fraud-rate", type=float, default=0.06)
    p.add_argument("--business-type", default="bank", choices=[b.value for b in BusinessType])
    p.add_argument("--risk-appetite", default="medium", choices=[r.value for r in RiskAppetite])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--model", default=DEFAULT_MODEL_PATH)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    if not os.path.exists(args.model):
        raise SystemExit(
            f"Champion model not found at {args.model}. Train it first:\n"
            "  python scripts/train_champion_model.py"
        )

    df = _load_input(args)
    predictor = ChampionPredictor.load(args.model)
    scored = predictor.predict_batch(df)
    _write(scored, args.out)

    counts = scored["decision"].value_counts().to_dict()
    print(f"scored {len(scored)} transactions -> {args.out}")
    print(f"decision breakdown: {counts}")
    print(f"mean fraud_probability: {scored['fraud_probability'].mean():.4f}")
    print("\nexample response records:")
    for rec in predictor.predict_response(df.head(3)):
        print(f"  {rec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
