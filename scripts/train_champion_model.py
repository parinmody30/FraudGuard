#!/usr/bin/env python
"""Train and persist the FraudLab champion model (XGBoost) + report (Phase 4).

Generates a simulator dataset (Phase 3 config), trains the champion with the exact Phase 3
preprocessing and hyperparameters, evaluates on held-out users, persists the bundle under
``models/``, and writes ``reports/champion_model_report.md``.

Example:
    python scripts/train_champion_model.py --num-users 400 --fraud-rate 0.06 --seed 42
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

from fraudlab.config.config import BusinessType, FraudLabConfig, RiskAppetite  # noqa: E402
from fraudlab.exporters.export import transactions_to_dataframe  # noqa: E402
from fraudlab.models.champion import train_champion  # noqa: E402
from fraudlab.models.persistence import (  # noqa: E402
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    save_bundle,
)
from fraudlab.pipeline import build_dataset  # noqa: E402


def _report(meta: dict, paths: dict) -> str:
    m = meta["metrics"]
    th = meta["thresholds"]
    cb = meta["confidence_bands"]
    L = [
        "# FraudLab Champion Model Report\n",
        "## Champion Model\n",
        "**XGBoost** (XGBoostClassifier) — selected in Phase 3 (PR-AUC 0.903) and confirmed "
        "as the sole production detector in Phase 3.1 (no ensemble).\n",
        "## Metrics (held-out users)\n",
        "| metric | value |",
        "|---|---|",
        f"| PR-AUC | {m['pr_auc']} |",
        f"| ROC-AUC | {m['roc_auc']} |",
        f"| Precision | {m['precision']} |",
        f"| Recall | {m['recall']} |",
        f"| F1 | {m['f1']} |",
        f"| Recall_hard | {m['recall_hard']} |",
        f"| Recall_easy | {m['recall_easy']} |",
        f"| Recall_medium | {m['recall_medium']} |",
        f"| Recall@1% / 2% / 5% | {m['recall_at_1pct']} / {m['recall_at_2pct']} / {m['recall_at_5pct']} |",
        "",
        "## Decision thresholds\n",
        f"- ALLOW: probability < {th['review']}",
        f"- REVIEW: {th['review']} ≤ probability < {th['block']}",
        f"- BLOCK: probability ≥ {th['block']}",
        "",
        "## Confidence bands\n",
        f"- LOW: probability < {cb['medium']}",
        f"- MEDIUM: {cb['medium']} ≤ probability ≤ {cb['high']}",
        f"- HIGH: probability > {cb['high']}",
        "",
        "## Dataset\n",
        f"- Simulator config: {meta['simulator_config']}",
        f"- Rows: {meta['dataset_size']}, users: {meta['n_users']}, fraud rate: {meta['fraud_rate']}",
        f"- Split: {meta['split']}",
        f"- Features after preprocessing: {meta['n_features']}",
        "",
        "## Artifacts\n",
        f"- Model bundle: `{paths['model']}`",
        f"- Metadata: `{paths['metadata']}`",
        f"- Training timestamp: {meta['training_timestamp']}",
        "",
        "## Recommendation\n",
        "**Champion model approved for explainability integration.**",
    ]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train the FraudLab champion model.")
    p.add_argument("--num-users", type=int, default=400)
    p.add_argument("--transactions-per-user", type=int, default=50)
    p.add_argument("--fraud-rate", type=float, default=0.06)
    p.add_argument("--business-type", default="bank", choices=[b.value for b in BusinessType])
    p.add_argument("--risk-appetite", default="medium", choices=[r.value for r in RiskAppetite])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--model-out", default=DEFAULT_MODEL_PATH)
    p.add_argument("--metadata-out", default=DEFAULT_METADATA_PATH)
    p.add_argument("--report-out", default=os.path.join("reports", "champion_model_report.md"))
    args = p.parse_args(argv)

    config = FraudLabConfig(
        num_users=args.num_users,
        transactions_per_user=args.transactions_per_user,
        fraud_rate=args.fraud_rate,
        business_type=BusinessType(args.business_type),
        risk_appetite=RiskAppetite(args.risk_appetite),
        random_seed=args.seed,
    )
    sim_config = {
        "num_users": config.num_users,
        "transactions_per_user": config.transactions_per_user,
        "fraud_rate": config.fraud_rate,
        "business_type": config.business_type.value,
        "risk_appetite": config.risk_appetite.value,
        "seed": config.random_seed,
    }
    transactions, _ = build_dataset(config)
    df = transactions_to_dataframe(transactions)

    bundle, _metrics = train_champion(
        df, simulator_config=sim_config, test_size=args.test_size, random_state=args.seed
    )
    paths = save_bundle(bundle, model_path=args.model_out, metadata_path=args.metadata_out)

    os.makedirs(os.path.dirname(os.path.abspath(args.report_out)), exist_ok=True)
    report = _report(bundle.metadata, paths)
    with open(args.report_out, "w", encoding="utf-8") as fh:
        fh.write(report)

    print(report)
    print("\nTraining summary:")
    print(json.dumps({"artifacts": paths, "metrics": bundle.metadata["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
