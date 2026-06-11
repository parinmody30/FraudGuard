"""Merchant model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Merchant:
    """A merchant a user can transact with.

    ``merchant_risk_score`` (0..1) correlates with the category (e.g. crypto/gambling
    are riskier than groceries) plus noise, so it is a usable — but not deterministic —
    signal for downstream detectors.
    """

    merchant_id: str
    merchant_name: str
    merchant_category: str
    merchant_risk_score: float
    city: str
    country: str
    supported_payment_methods: list[str]
