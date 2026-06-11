#!/usr/bin/env python
"""Demo: build a hybrid dataset (simulator hard fraud + CTGAN soft rows) from a trained model.

Loads the persisted CTGAN, regenerates the matching simulator dataset (for its hard-fraud
pool), merges them into a hybrid dataset with a requested fraud rate/mix, writes it out, and
prints the composition. Used to produce the Phase 2 hybrid-composition figures.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fraudlab.config.config import FraudLabConfig, RiskAppetite, default_fraud_mix
from fraudlab.exporters.export import transactions_to_dataframe
from fraudlab.gan.merger import build_hybrid_dataset
from fraudlab.gan.sampler import CTGANSampler
from fraudlab.gan.trainer import load_artifacts
from fraudlab.pipeline import build_dataset


def main() -> int:
    config = FraudLabConfig(
        num_users=250, transactions_per_user=50, fraud_rate=0.08,
        risk_appetite=RiskAppetite.MEDIUM, random_seed=42,
    )
    transactions, _ = build_dataset(config)
    sim_df = transactions_to_dataframe(transactions)

    model, metadata, _ = load_artifacts()
    sampler = CTGANSampler(model, metadata, seed=42)

    mix = default_fraud_mix(RiskAppetite.MEDIUM)
    hybrid, comp = build_hybrid_dataset(
        sim_df, sampler, n_rows=20000, fraud_rate=0.05, fraud_mix=mix, seed=42
    )

    os.makedirs("out", exist_ok=True)
    hybrid.to_csv("out/hybrid_dataset.csv", index=False)
    print(json.dumps(comp, indent=2))
    print("\nwritten: out/hybrid_dataset.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
