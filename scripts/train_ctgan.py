#!/usr/bin/env python
"""Train a CTGAN realism augmenter on the soft (legit + easy + medium) transaction rows.

The simulator dataset is loaded (or generated), reduced to the CTGAN training frame (hard
fraud excluded, identifiers dropped), and used to fit a CTGAN. The trained model, its
metadata, and the training config are persisted under ``models/``.

Examples
--------
    python scripts/train_ctgan.py --generate --num-users 200 --transactions-per-user 60 \
        --fraud-rate 0.08 --epochs 120 --seed 42
    python scripts/train_ctgan.py --input out/bank_seed42.csv --epochs 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _gan_common import add_source_args, resolve_dataframe  # noqa: E402

from fraudlab.gan.dataset_builder import build_training_frame, training_composition  # noqa: E402
from fraudlab.gan.metadata import infer_metadata  # noqa: E402
from fraudlab.gan.trainer import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    CTGANConfig,
    save_artifacts,
    train_ctgan,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a CTGAN realism augmenter.")
    add_source_args(parser)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--generator-dim", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--discriminator-dim", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--pac", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-log-frequency",
        dest="log_frequency",
        action="store_false",
        help="Disable CTGAN log-frequency conditional sampling (reduces minority oversampling).",
    )
    parser.set_defaults(log_frequency=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--model-out", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-out", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--config-out", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)

    sim_df, _ = resolve_dataframe(args)

    composition = training_composition(sim_df)
    print("training-set composition:")
    print(json.dumps(composition, indent=2))

    training_frame = build_training_frame(sim_df)
    metadata = infer_metadata(sim_df)

    config = CTGANConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        generator_dim=tuple(args.generator_dim),
        discriminator_dim=tuple(args.discriminator_dim),
        pac=args.pac,
        random_seed=args.seed,
        log_frequency=args.log_frequency,
        verbose=args.verbose,
    )

    print(f"\ntraining CTGAN on {len(training_frame)} rows for {config.epochs} epochs ...")
    model = train_ctgan(training_frame, metadata, config)

    paths = save_artifacts(
        model,
        metadata,
        config,
        model_path=args.model_out,
        metadata_path=args.metadata_out,
        config_path=args.config_out,
    )
    print("\nartifacts written:")
    print(json.dumps(paths, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
