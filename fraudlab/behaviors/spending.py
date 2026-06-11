"""Spending behavior: amount distribution, salary cycle, account maturation."""

from __future__ import annotations

import numpy as np

from fraudlab.config import reference_data as rd
from fraudlab.config.config import FraudLabConfig
from fraudlab.models import User


def maturation_factor(account_age_days: int) -> float:
    """Spending grows slowly as an account matures, saturating over a few years.

    Returns a multiplier in roughly [0.6, 1.2].
    """
    years = account_age_days / 365.0
    return float(0.6 + 0.6 * (1.0 - np.exp(-years / 2.5)))


def sample_amount(
    user: User,
    category: str,
    account_age_days: int,
    rng: np.random.Generator,
    config: FraudLabConfig,
) -> float:
    """Sample a transaction amount (INR) for a user/category.

    Base draw is the user's personal lognormal spend, scaled by the category multiplier,
    account maturation, and config noise. This produces heavy-tailed, per-user-distinct
    amount distributions rather than a homogeneous population.
    """
    base = float(rng.lognormal(mean=user.spend_mu, sigma=user.spend_sigma))
    _risk, cat_mult = rd.MERCHANT_CATEGORIES.get(category, (0.1, 1.0))
    amount = base * cat_mult * maturation_factor(account_age_days)

    # Multiplicative noise driven by config.noise_level keeps amounts varied.
    if config.noise_level > 0:
        jitter = rng.normal(1.0, 0.15 * config.noise_level)
        amount *= max(0.2, jitter)

    return round(max(10.0, amount), 2)


def salary_amount(user: User, rng: np.random.Generator) -> float:
    lo, hi = user.salary_range
    return round(float(rng.uniform(lo, hi)), 2)
