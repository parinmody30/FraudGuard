"""Baseline transaction generator.

Produces realistic *legitimate* transactions per user by composing the behavior modules
(spending, temporal, location, velocity) plus recurring expenses and device/merchant
selection. Output transactions all carry ``is_fraud=False``; the fraud engine mutates a
subset afterward.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

from fraudlab.behaviors.location import plan_travel_days, sample_location
from fraudlab.behaviors.spending import sample_amount
from fraudlab.behaviors.temporal import sample_timestamp, timeline_days
from fraudlab.behaviors.velocity import RollingFeatures
from fraudlab.config.config import FraudLabConfig
from fraudlab.generators.ids import IdMinter
from fraudlab.generators.rng import per_index_rng
from fraudlab.models import Beneficiary, Device, Merchant, Transaction, User

# Categories used for stable monthly recurring expenses.
_RECURRING_CATEGORIES = ("utilities", "telecom", "entertainment", "education")


class BaselineTransactionGenerator:
    """Generate legitimate transactions for the whole population."""

    def __init__(
        self,
        config: FraudLabConfig,
        merchants: list[Merchant],
        devices: dict[str, list[Device]],
        beneficiaries: dict[str, list[Beneficiary]],
        id_minter: IdMinter | None = None,
    ) -> None:
        self.config = config
        self.merchants = merchants
        self.devices = devices
        self.beneficiaries = beneficiaries
        self._days = timeline_days(config)
        self._by_category = self._index_merchants(merchants)
        self._ids = id_minter or IdMinter()

    @staticmethod
    def _index_merchants(merchants: list[Merchant]) -> dict[str, list[Merchant]]:
        idx: dict[str, list[Merchant]] = defaultdict(list)
        for m in merchants:
            idx[m.merchant_category].append(m)
        return idx

    def generate(self, users: list[User]) -> list[Transaction]:
        all_txns: list[Transaction] = []
        for u_idx, user in enumerate(users):
            rng = per_index_rng(self.config.random_seed, "baseline", u_idx)
            user_txns = self._generate_for_user(user, u_idx, rng)
            RollingFeatures.apply(user_txns)
            all_txns.extend(user_txns)
        return all_txns

    # --------------------------------------------------------------- per-user

    def _generate_for_user(
        self, user: User, u_idx: int, rng: np.random.Generator
    ) -> list[Transaction]:
        n = self.config.transactions_per_user
        travel_map = plan_travel_days(user, self._days, rng)
        user_devices = self.devices[user.user_id]

        txns: list[Transaction] = []

        # 1) Recurring monthly expenses (stable category + amount + day-of-month).
        recurring = self._plan_recurring(user, rng)
        for cat, amount, dom, method in recurring:
            for ts in self._monthly_dates(dom):
                if len(txns) >= n:
                    break
                txns.append(self._build_txn(user, ts, rng, category=cat, amount=amount, payment_method=method, travel_map=travel_map, devices=user_devices))

        # 2) Discretionary transactions fill the remainder.
        remaining = max(0, n - len(txns))
        if remaining:
            timestamps = sample_timestamp(user, rng, self.config, n=remaining)
            for ts in timestamps:
                txns.append(self._build_txn(user, ts, rng, category=None, amount=None, payment_method=None, travel_map=travel_map, devices=user_devices))

        return txns

    def _plan_recurring(
        self, user: User, rng: np.random.Generator
    ) -> list[tuple[str, float, int, str]]:
        k = int(np.clip(rng.poisson(2) + 1, 1, 4))
        items: list[tuple[str, float, int, str]] = []
        for _ in range(k):
            cat = str(rng.choice(_RECURRING_CATEGORIES))
            amount = sample_amount(user, cat, user.account_age_days, rng, self.config)
            dom = int(rng.integers(1, 28))
            method = str(rng.choice(user.preferred_payment_methods))
            items.append((cat, round(amount, 2), dom, method))
        return items

    def _monthly_dates(self, day_of_month: int) -> list[datetime]:
        out: list[datetime] = []
        start = self.config.start_date
        # Walk month by month across the timeline window.
        year, month = start.year, start.month
        for _ in range(self._days // 28 + 2):
            try:
                d = start.replace(year=year, month=month, day=day_of_month)
            except ValueError:
                d = start.replace(year=year, month=month, day=28)
            offset = (d - start).days
            if 0 <= offset < self._days:
                out.append(datetime.combine(d, datetime.min.time()) + timedelta(hours=10))
            month += 1
            if month > 12:
                month = 1
                year += 1
        return out

    # ------------------------------------------------------------- txn builder

    def _build_txn(
        self,
        user: User,
        ts: datetime,
        rng: np.random.Generator,
        *,
        category: str | None,
        amount: float | None,
        payment_method: str | None,
        travel_map: dict[int, tuple[str, str, str]],
        devices: list[Device],
    ) -> Transaction:
        day_offset = (ts.date() - self.config.start_date).days
        day_offset = max(0, min(self._days - 1, day_offset))

        if category is None:
            category = self._pick_category(user, rng)
        merchant = self._pick_merchant(category, rng)
        if payment_method is None:
            payment_method = self._pick_payment_method(user, merchant, rng)
        channel = str(rng.choice(self.config.channels))

        device = self._pick_device(devices, day_offset, rng)
        city, state, country = sample_location(user, day_offset, travel_map)
        account_age = user.account_age_days + day_offset

        if amount is None:
            amount = sample_amount(user, category, account_age, rng, self.config)

        return Transaction(
            transaction_id=self._ids.next(),
            user_id=user.user_id,
            timestamp=ts,
            amount=amount,
            currency=self.config.currency,
            merchant_id=merchant.merchant_id,
            merchant_name=merchant.merchant_name,
            merchant_category=merchant.merchant_category,
            payment_method=payment_method,
            channel=channel,
            device_id=device.device_id,
            device_type=device.device_type,
            city=city,
            state=state,
            country=country,
            is_foreign=country != user.country,
            account_age_days=account_age,
            avg_user_spend=0.0,   # filled by RollingFeatures
            median_user_spend=0.0,
            velocity_1h=0,
            velocity_24h=0,
            merchant_risk_score=merchant.merchant_risk_score,
            device_risk_score=round(1.0 - device.trust_score, 4),
            is_fraud=False,
            fraud_type=None,
        )

    # ------------------------------------------------------------- selectors

    def _pick_category(self, user: User, rng: np.random.Generator) -> str:
        # 70% from preferred categories, 30% exploratory across the full catalog.
        if rng.random() < 0.7 and user.preferred_merchant_categories:
            return str(rng.choice(user.preferred_merchant_categories))
        return str(rng.choice(list(self._by_category.keys())))

    def _pick_merchant(self, category: str, rng: np.random.Generator) -> Merchant:
        candidates = self._by_category.get(category)
        if not candidates:
            # Fall back to any merchant if the pool lacks this category.
            candidates = self.merchants
        return candidates[int(rng.integers(len(candidates)))]

    def _pick_payment_method(
        self, user: User, merchant: Merchant, rng: np.random.Generator
    ) -> str:
        usable = [m for m in user.preferred_payment_methods if m in merchant.supported_payment_methods]
        if not usable:
            usable = merchant.supported_payment_methods or list(self.config.payment_methods)
        return str(rng.choice(usable))

    def _pick_device(
        self, devices: list[Device], day_offset: int, rng: np.random.Generator
    ) -> Device:
        available = [d for d in devices if d.first_seen_day <= day_offset]
        if not available:
            available = [devices[0]]
        # Weight by trust so the primary/established device dominates usage.
        weights = np.array([max(0.05, d.trust_score) for d in available])
        weights /= weights.sum()
        return available[int(rng.choice(len(available), p=weights))]
