#!/usr/bin/env python
"""Evaluate a trained CTGAN against the simulator's soft-row distribution.

Loads (or generates) a simulator dataset, restricts it to the soft rows CTGAN was trained
on (legit + easy + medium, identifiers dropped), samples an equal number of synthetic rows,
and computes statistical-fidelity + ML-utility (TSTR/TRTS) metrics. Writes:

    reports/ctgan_evaluation.json
    reports/ctgan_evaluation.md
    reports/ctgan_similarity.csv

and prints the READY-FOR-HYBRID / NEEDS-TUNING recommendation.

Example
-------
    python scripts/evaluate_ctgan.py --input out/bank_seed42.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _gan_common import add_source_args, resolve_dataframe  # noqa: E402

from fraudlab.gan.dataset_builder import HARD_FRAUD_TYPES  # noqa: E402
from fraudlab.gan.evaluator import (  # noqa: E402
    evaluate_ctgan,
    format_markdown,
    similarity_table,
)
from fraudlab.gan.metadata import DROP_COLUMNS  # noqa: E402
from fraudlab.gan.sampler import CTGANSampler  # noqa: E402
from fraudlab.gan.trainer import (  # noqa: E402
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    load_artifacts,
)

REPORTS_DIR = "reports"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained CTGAN.")
    add_source_args(parser)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reports-dir", default=REPORTS_DIR)
    args = parser.parse_args(argv)

    sim_df, _ = resolve_dataframe(args)
    model, metadata, _ = load_artifacts(model_path=args.model, metadata_path=args.metadata)
    sampler = CTGANSampler(model, metadata, seed=args.seed)

    # Real reference = simulator soft rows, identifiers dropped, fraud_type None on legit.
    real = sim_df[~sim_df["fraud_type"].isin(HARD_FRAUD_TYPES)].drop(
        columns=[c for c in DROP_COLUMNS if c in sim_df.columns]
    ).reset_index(drop=True)

    # Synthetic = same count, sampled unconditionally so we measure CTGAN's intrinsic
    # prevalence / type-mix / marginal fidelity.
    synth = sampler.generate(len(real), seed=args.seed)

    report = evaluate_ctgan(real, synth, metadata, random_state=args.seed)

    os.makedirs(args.reports_dir, exist_ok=True)
    json_path = os.path.join(args.reports_dir, "ctgan_evaluation.json")
    md_path = os.path.join(args.reports_dir, "ctgan_evaluation.md")
    csv_path = os.path.join(args.reports_dir, "ctgan_similarity.csv")

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(format_markdown(report))
    similarity_table(report["similarity"]).to_csv(csv_path, index=False)

    print(format_markdown(report))
    print(f"\nwritten:\n  {json_path}\n  {md_path}\n  {csv_path}")
    print(f"\nRECOMMENDATION: {report['recommendation']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
