"""End-to-end generation pipeline tying all subsystems together."""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

from fraudlab.config.config import FraudLabConfig
from fraudlab.fraud.engine import FraudEngine
from fraudlab.generators import (
    BaselineTransactionGenerator,
    BeneficiaryGenerator,
    DeviceGenerator,
    IdMinter,
    MerchantGenerator,
    PopulationGenerator,
)
from fraudlab.models import Transaction


def build_dataset(config: FraudLabConfig) -> tuple[list[Transaction], dict[str, Any]]:
    """Generate a complete synthetic dataset.

    Returns the list of transactions and a summary dict. Generation is fully
    deterministic for a given ``config`` (including ``random_seed``).
    """
    start = time.perf_counter()

    # 1) Population and accounts.
    users, _accounts = PopulationGenerator(config).generate()

    # 2) Shared entities.
    merchants = MerchantGenerator(config).generate()
    devices = DeviceGenerator(config).generate(users)
    beneficiaries = BeneficiaryGenerator(config).generate(users)

    # 3) Baseline (legitimate) transactions. Shared id minter keeps ids globally unique.
    id_minter = IdMinter(start=0)
    baseline = BaselineTransactionGenerator(
        config, merchants, devices, beneficiaries, id_minter=id_minter
    ).generate(users)

    # 4) Fraud injection (net additions on top of the baseline).
    engine = FraudEngine(config, users, merchants, devices, beneficiaries, id_minter)
    transactions = engine.inject(baseline)

    elapsed = time.perf_counter() - start
    summary = _summarize(config, transactions, elapsed)
    return transactions, summary


def _summarize(
    config: FraudLabConfig, transactions: list[Transaction], elapsed: float
) -> dict[str, Any]:
    total = len(transactions)
    fraud_txns = [t for t in transactions if t.is_fraud]
    fraud_count = len(fraud_txns)
    fraud_dist = dict(Counter(t.fraud_type for t in fraud_txns))
    return {
        "total_transactions": total,
        "fraud_count": fraud_count,
        "fraud_percentage": round(100.0 * fraud_count / total, 4) if total else 0.0,
        "fraud_type_distribution": fraud_dist,
        "business_type": config.business_type.value,
        "generation_time_seconds": round(elapsed, 3),
        "random_seed": config.random_seed,
        "num_users": config.num_users,
        "transactions_per_user": config.transactions_per_user,
    }
