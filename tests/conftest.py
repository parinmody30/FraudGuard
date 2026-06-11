"""Shared pytest fixtures: small, fast configs and a cached dataset."""

from __future__ import annotations

import os
import sys

import pytest

# Ensure the package is importable when tests run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fraudlab.config.config import BusinessType, FraudLabConfig, RiskAppetite  # noqa: E402
from fraudlab.pipeline import build_dataset  # noqa: E402


@pytest.fixture
def small_config() -> FraudLabConfig:
    return FraudLabConfig(
        num_users=40,
        transactions_per_user=40,
        fraud_rate=0.05,
        business_type=BusinessType.BANK,
        risk_appetite=RiskAppetite.MEDIUM,
        noise_level=0.3,
        random_seed=123,
    )


@pytest.fixture
def small_dataset(small_config):
    transactions, summary = build_dataset(small_config)
    return transactions, summary, small_config
