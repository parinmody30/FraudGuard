"""Fraud rate and fraud mix correctness."""

from __future__ import annotations

from collections import Counter

from fraudlab.config.config import (
    ALL_FRAUD_TYPES,
    BusinessType,
    FraudLabConfig,
    RiskAppetite,
)
from fraudlab.pipeline import build_dataset


def _expected_target(config: FraudLabConfig) -> int:
    b = config.num_users * config.transactions_per_user
    r = config.fraud_rate
    return round(b * r / (1.0 - r))


def test_baseline_count_is_exact(small_dataset):
    transactions, _, config = small_dataset
    legit = [t for t in transactions if not t.is_fraud]
    assert len(legit) == config.num_users * config.transactions_per_user


def test_realized_fraud_rate_matches(small_dataset):
    transactions, summary, config = small_dataset
    realized = summary["fraud_count"] / summary["total_transactions"]
    assert abs(realized - config.fraud_rate) < 0.005


def test_fraud_count_matches_target(small_dataset):
    _, summary, config = small_dataset
    assert summary["fraud_count"] == _expected_target(config)


def test_default_mix_covers_all_types(small_dataset):
    transactions, _, _ = small_dataset
    present = {t.fraud_type for t in transactions if t.is_fraud}
    # The default medium mix assigns positive weight to every scenario.
    assert present == set(ALL_FRAUD_TYPES)


def test_custom_mix_only_uses_specified_types():
    config = FraudLabConfig(
        num_users=60,
        transactions_per_user=50,
        fraud_rate=0.06,
        business_type=BusinessType.CREDIT_CARD,
        risk_appetite=RiskAppetite.HIGH,
        random_seed=99,
        fraud_mix={"large_spend": 0.5, "card_testing": 0.3, "gradual_escalation": 0.2},
    )
    transactions, summary = build_dataset(config)
    counts = Counter(t.fraud_type for t in transactions if t.is_fraud)
    assert set(counts) <= {"large_spend", "card_testing", "gradual_escalation"}
    # Proportions should be in the right ballpark.
    total = sum(counts.values())
    assert counts["large_spend"] / total > counts["gradual_escalation"] / total
