#!/usr/bin/env python
"""Sample synthetic rows from a trained CTGAN.

Modes (mutually exclusive):
* default            - unconditional ``generate(n)``.
* ``--fraud-type T`` - conditional sampling of a specific easy/medium scenario (or
                       ``--fraud-type legit`` for legitimate rows).
* ``--fraud-rate R`` - prevalence-controlled batch hitting fraud rate R.

Examples
--------
    python scripts/sample_ctgan.py --n 1000 --out out/ctgan_sample.csv
    python scripts/sample_ctgan.py --n 500 --fraud-type large_spend --out out/ls.csv
    python scripts/sample_ctgan.py --n 2000 --fraud-rate 0.05 --out out/mixed.csv
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fraudlab.gan.dataset_builder import LEGIT_SENTINEL  # noqa: E402
from fraudlab.gan.sampler import CTGANSampler  # noqa: E402
from fraudlab.gan.trainer import (  # noqa: E402
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    load_artifacts,
)


def _write(df, path: str) -> None:
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
    parser = argparse.ArgumentParser(description="Sample from a trained CTGAN.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--fraud-type", help="Scenario name, or 'legit' for legitimate rows.")
    parser.add_argument("--fraud-rate", type=float, help="Prevalence-controlled sampling.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    model, metadata, _ = load_artifacts(model_path=args.model, metadata_path=args.metadata)
    sampler = CTGANSampler(model, metadata, seed=args.seed)

    if args.fraud_type is not None:
        ftype = None if args.fraud_type == LEGIT_SENTINEL else args.fraud_type
        df = sampler.conditional_generate(ftype, args.n, seed=args.seed)
    elif args.fraud_rate is not None:
        df = sampler.sample_with_prevalence(args.n, fraud_rate=args.fraud_rate, seed=args.seed)
    else:
        df = sampler.generate(args.n, seed=args.seed)

    _write(df, args.out)
    print(f"sampled {len(df)} rows -> {args.out}")
    print(f"fraud rows: {int(df['is_fraud'].sum())} ({df['is_fraud'].mean():.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
