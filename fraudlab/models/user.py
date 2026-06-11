"""User and Account models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(slots=True)
class User:
    """A synthetic customer with a distinct, stable behavioral profile.

    The behavioral parameters (``spend_mu``/``spend_sigma``, ``typical_active_hours``,
    ``travel_frequency``, ``device_churn_rate`` and the preference lists) are what make
    one user differ meaningfully from another, driving population diversity.
    """

    user_id: str
    age: int
    occupation: str
    income_band: str
    salary_range: tuple[int, int]
    city: str
    state: str
    country: str
    account_age_days: int

    preferred_payment_methods: list[str]
    preferred_merchant_categories: list[str]
    typical_active_hours: list[int]

    travel_frequency: float          # 0..1 propensity to transact away from home
    device_churn_rate: float         # 0..1 propensity to acquire new devices over time

    # Per-user lognormal spend parameters (log-space mean / sigma). Sampled across a
    # wide range so spending magnitudes and dispersions differ strongly between users.
    spend_mu: float
    spend_sigma: float

    # Day-of-month (1..28) on which salary is typically credited.
    salary_day: int = 1

    # Behavioral archetype label, useful for diversity diagnostics/tests.
    archetype: str = "regular"


@dataclass(slots=True)
class Account:
    """A financial account owned by a user, consistent with the dataset business type."""

    account_id: str
    user_id: str
    business_type: str
    opened_date: date
    balance_band: str
    status: str = "active"
