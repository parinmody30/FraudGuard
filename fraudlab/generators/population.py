"""Population generator: realistic, heterogeneous users and their accounts."""

from __future__ import annotations

from datetime import timedelta

import numpy as np

from fraudlab.config import reference_data as rd
from fraudlab.config.config import FraudLabConfig
from fraudlab.generators.rng import per_index_rng, stream_rng
from fraudlab.models import Account, User


def _weighted_choice(rng: np.random.Generator, options: dict[str, float]) -> str:
    keys = list(options.keys())
    weights = np.array(list(options.values()), dtype=float)
    weights /= weights.sum()
    return str(rng.choice(keys, p=weights))


class PopulationGenerator:
    """Generate ``num_users`` users with deliberately diverse behavioral profiles."""

    def __init__(self, config: FraudLabConfig) -> None:
        self.config = config

    def generate(self) -> tuple[list[User], dict[str, Account]]:
        users: list[User] = []
        accounts: dict[str, Account] = {}
        for i in range(self.config.num_users):
            rng = per_index_rng(self.config.random_seed, "population", i)
            user = self._make_user(i, rng)
            users.append(user)
            accounts[user.user_id] = self._make_account(user, rng)
        return users, accounts

    # ------------------------------------------------------------------ helpers

    def _make_user(self, index: int, rng: np.random.Generator) -> User:
        user_id = f"U{index:07d}"

        occupation = str(rng.choice(rd.OCCUPATIONS))
        income_band = _weighted_choice(rng, rd.OCCUPATION_INCOME_WEIGHTS[occupation])
        salary_lo, salary_hi = rd.INCOME_BAND_SALARY[income_band]
        salary = int(rng.integers(salary_lo, salary_hi + 1))
        salary_range = (int(salary * 0.9), int(salary * 1.1))

        age = self._sample_age(occupation, rng)
        city, state = rd.INDIA_CITIES[int(rng.integers(len(rd.INDIA_CITIES)))]
        account_age_days = int(rng.integers(15, 12 * 365))

        # Behavioral spread: per-user lognormal spend parameters anchored on salary so a
        # user's typical transaction scales with income, with wide per-user dispersion.
        # spend_mu is in log-space; derive from a fraction of monthly salary.
        typical_txn = max(150.0, salary * float(rng.uniform(0.01, 0.06)))
        spend_mu = float(np.log(typical_txn))
        spend_sigma = float(rng.uniform(0.35, 1.1))

        archetype = str(rng.choice(list(rd.ACTIVE_HOUR_ARCHETYPES.keys())))
        active_hours = list(rd.ACTIVE_HOUR_ARCHETYPES[archetype])

        n_methods = int(rng.integers(1, len(self.config.payment_methods) + 1))
        preferred_payment_methods = list(
            rng.choice(self.config.payment_methods, size=n_methods, replace=False)
        )

        categories = list(rd.MERCHANT_CATEGORIES.keys())
        n_cats = int(rng.integers(2, 5))
        preferred_merchant_categories = list(
            rng.choice(categories, size=n_cats, replace=False)
        )

        travel_frequency = float(np.clip(rng.beta(1.5, 8.0), 0.0, 1.0))
        device_churn_rate = float(np.clip(rng.beta(1.3, 9.0), 0.0, 1.0))
        salary_day = int(rng.integers(1, 29))

        return User(
            user_id=user_id,
            age=age,
            occupation=occupation,
            income_band=income_band,
            salary_range=salary_range,
            city=city,
            state=state,
            country=rd.HOME_COUNTRY,
            account_age_days=account_age_days,
            preferred_payment_methods=preferred_payment_methods,
            preferred_merchant_categories=preferred_merchant_categories,
            typical_active_hours=active_hours,
            travel_frequency=travel_frequency,
            device_churn_rate=device_churn_rate,
            spend_mu=spend_mu,
            spend_sigma=spend_sigma,
            salary_day=salary_day,
            archetype=archetype,
        )

    def _sample_age(self, occupation: str, rng: np.random.Generator) -> int:
        if occupation == "student":
            return int(rng.integers(18, 26))
        if occupation == "retired":
            return int(rng.integers(60, 82))
        return int(np.clip(rng.normal(38, 11), 21, 70))

    def _make_account(self, user: User, rng: np.random.Generator) -> Account:
        opened = self.config.start_date - timedelta(days=user.account_age_days)
        balance_band = user.income_band
        return Account(
            account_id=f"A{user.user_id[1:]}",
            user_id=user.user_id,
            business_type=self.config.business_type.value,
            opened_date=opened,
            balance_band=balance_band,
            status="active",
        )
