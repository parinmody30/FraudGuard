"""Schema correctness of generated transactions."""

from __future__ import annotations

from datetime import datetime

from fraudlab.config.config import BUSINESS_CHANNELS, BUSINESS_PAYMENT_METHODS
from fraudlab.models import TRANSACTION_FIELDS


def test_all_fields_present(small_dataset):
    transactions, _, _ = small_dataset
    for t in transactions[:200]:
        row = t.to_row()
        assert tuple(row.keys()) == TRANSACTION_FIELDS


def test_field_types(small_dataset):
    transactions, _, _ = small_dataset
    for t in transactions[:200]:
        assert isinstance(t.transaction_id, str)
        assert isinstance(t.user_id, str)
        assert isinstance(t.timestamp, datetime)
        assert isinstance(t.amount, float) and t.amount > 0
        assert isinstance(t.is_foreign, bool)
        assert isinstance(t.is_fraud, bool)
        assert isinstance(t.velocity_1h, int) and t.velocity_1h >= 1
        assert isinstance(t.velocity_24h, int) and t.velocity_24h >= 1
        assert 0.0 <= t.merchant_risk_score <= 1.0
        assert 0.0 <= t.device_risk_score <= 1.0
        assert (t.fraud_type is None) == (not t.is_fraud)


def test_is_foreign_consistency(small_dataset):
    transactions, _, _ = small_dataset
    for t in transactions:
        assert t.is_foreign == (t.country != "India")


def test_payment_method_and_channel_vocab(small_dataset):
    transactions, _, config = small_dataset
    methods = set(BUSINESS_PAYMENT_METHODS[config.business_type])
    channels = set(BUSINESS_CHANNELS[config.business_type])
    for t in transactions:
        assert t.payment_method in methods
        assert t.channel in channels
