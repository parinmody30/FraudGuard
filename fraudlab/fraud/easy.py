"""Easy fraud scenarios.

These are deliberately detectable by single-row signals (amount spike, new device,
impossible geography, odd hour) — but note that ``new_device`` keeps amounts normal and
legitimate baseline activity also includes new devices and travel, so even "easy" fraud
is not perfectly separable by one naive rule.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np

from fraudlab.fraud.base import (
    FraudContext,
    anchor_time,
    minutes,
    register,
)
from fraudlab.models import Transaction, User


def _clamp(budget: int, desired: int) -> int:
    return max(1, min(budget, desired))


@register
class AccountTakeover:
    name = "account_takeover"
    difficulty = "easy"

    def inject(self, user: User, user_txns: list[Transaction], ctx: FraudContext,
               rng: np.random.Generator, budget: int) -> list[Transaction]:
        k = _clamp(budget, int(rng.integers(3, 8)))
        base = ctx.typical_amount(user_txns, user)
        start = anchor_time(user_txns, rng) + timedelta(hours=float(rng.uniform(1, 8)))
        device = ctx.new_device(user, ctx.day_offset(start), rng)
        # Sometimes the takeover happens from an unusual location, sometimes from home.
        if rng.random() < 0.5:
            location = ctx.foreign_location(rng) if rng.random() < 0.5 else ctx.distant_domestic_location(user, rng)
        else:
            location = (user.city, user.state, user.country)

        out: list[Transaction] = []
        ts = start
        for i in range(k):
            amount = base * float(rng.uniform(2.0, 8.0)) * (1.0 + 0.2 * i)
            merchant = ctx.pick_merchant(rng, high_risk=rng.random() < 0.4)
            out.append(ctx.make_txn(user, ts, amount, self.name, merchant=merchant,
                                    device=device, location=location, rng=rng))
            ts = ts + minutes(rng, 3, 40)
        return out


@register
class LargeSpend:
    name = "large_spend"
    difficulty = "easy"

    def inject(self, user: User, user_txns: list[Transaction], ctx: FraudContext,
               rng: np.random.Generator, budget: int) -> list[Transaction]:
        k = _clamp(budget, int(rng.integers(1, 3)))
        base = ctx.typical_amount(user_txns, user)
        device = ctx.existing_device(user, rng)  # normal device — only amount is anomalous
        out: list[Transaction] = []
        for _ in range(k):
            ts = anchor_time(user_txns, rng) + minutes(rng, 5, 600)
            amount = base * float(rng.uniform(8.0, 20.0))
            merchant = ctx.pick_merchant(rng, category=str(rng.choice(["electronics", "jewelry", "travel"])))
            out.append(ctx.make_txn(user, ts, amount, self.name, merchant=merchant,
                                    device=device, rng=rng))
        return out


@register
class GeoAnomaly:
    name = "geo_anomaly"
    difficulty = "easy"

    def inject(self, user: User, user_txns: list[Transaction], ctx: FraudContext,
               rng: np.random.Generator, budget: int) -> list[Transaction]:
        k = _clamp(budget, int(rng.integers(1, 3)))
        base = ctx.typical_amount(user_txns, user)
        device = ctx.existing_device(user, rng)
        # Impossible travel: a foreign transaction minutes after home activity.
        anchor = anchor_time(user_txns, rng)
        location = ctx.foreign_location(rng)
        out: list[Transaction] = []
        ts = anchor + minutes(rng, 10, 90)
        for _ in range(k):
            amount = base * float(rng.uniform(0.8, 3.0))
            merchant = ctx.pick_merchant(rng)
            out.append(ctx.make_txn(user, ts, amount, self.name, merchant=merchant,
                                    device=device, location=location, rng=rng))
            ts = ts + minutes(rng, 15, 120)
        return out


@register
class NightTime:
    name = "night_time"
    difficulty = "easy"

    def inject(self, user: User, user_txns: list[Transaction], ctx: FraudContext,
               rng: np.random.Generator, budget: int) -> list[Transaction]:
        k = _clamp(budget, int(rng.integers(1, 4)))
        base = ctx.typical_amount(user_txns, user)
        device = ctx.existing_device(user, rng) if rng.random() < 0.6 else ctx.new_device(user, 0, rng)
        anchor = anchor_time(user_txns, rng)
        # Force a dead-of-night hour outside the user's typical active hours.
        night = anchor.replace(hour=int(rng.integers(2, 5)), minute=int(rng.integers(0, 60)))
        out: list[Transaction] = []
        ts = night
        for _ in range(k):
            amount = base * float(rng.uniform(1.5, 5.0))
            merchant = ctx.pick_merchant(rng, high_risk=rng.random() < 0.3)
            out.append(ctx.make_txn(user, ts, amount, self.name, merchant=merchant,
                                    device=device, rng=rng))
            ts = ts + minutes(rng, 5, 45)
        return out


@register
class NewDevice:
    name = "new_device"
    difficulty = "easy"

    def inject(self, user: User, user_txns: list[Transaction], ctx: FraudContext,
               rng: np.random.Generator, budget: int) -> list[Transaction]:
        k = _clamp(budget, int(rng.integers(1, 3)))
        base = ctx.typical_amount(user_txns, user)
        anchor = anchor_time(user_txns, rng)
        device = ctx.new_device(user, ctx.day_offset(anchor), rng)
        out: list[Transaction] = []
        ts = anchor + minutes(rng, 5, 300)
        for _ in range(k):
            # Amount stays within the user's normal range: the device is the only signal,
            # which alone is weak because legitimate device churn also exists.
            amount = base * float(rng.uniform(0.7, 1.6))
            merchant = ctx.pick_merchant(rng)
            out.append(ctx.make_txn(user, ts, amount, self.name, merchant=merchant,
                                    device=device, rng=rng))
            ts = ts + minutes(rng, 30, 600)
        return out
