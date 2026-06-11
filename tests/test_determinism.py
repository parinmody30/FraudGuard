"""Deterministic generation and seed reproducibility."""

from __future__ import annotations

from dataclasses import replace

from fraudlab.pipeline import build_dataset


def _signature(transactions):
    return [
        (t.transaction_id, t.user_id, t.timestamp, t.amount, t.merchant_id,
         t.device_id, t.is_fraud, t.fraud_type)
        for t in transactions
    ]


def test_same_seed_is_identical(small_config):
    a, _ = build_dataset(small_config)
    b, _ = build_dataset(small_config)
    assert _signature(a) == _signature(b)


def test_different_seed_differs(small_config):
    a, _ = build_dataset(small_config)
    other = replace(small_config, random_seed=small_config.random_seed + 1)
    b, _ = build_dataset(other)
    assert _signature(a) != _signature(b)


def test_counts_stable_across_runs(small_config):
    _, s1 = build_dataset(small_config)
    _, s2 = build_dataset(small_config)
    assert s1["total_transactions"] == s2["total_transactions"]
    assert s1["fraud_count"] == s2["fraud_count"]
    assert s1["fraud_type_distribution"] == s2["fraud_type_distribution"]
