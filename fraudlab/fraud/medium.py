"""Medium fraud scenarios.

These require simple aggregate context (velocity windows, merchant risk, beneficiary
recency) rather than a single field, and they keep individual amounts plausible.
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
class CardTesting:
    name = "card_testing"
    difficulty = "medium"

    def inject(self, user: User, user_txns: list[Transaction], ctx: FraudContext,
               rng: np.random.Generator, budget: int) -> list[Transaction]:
        # A burst of tiny authorizations across merchants within minutes.
        k = _clamp(budget, int(rng.integers(6, 16)))
        device = ctx.new_device(user, 0, rng) if rng.random() < 0.5 else ctx.existing_device(user, rng)
        ts = anchor_time(user_txns, rng) + minutes(rng, 1, 120)
        out: list[Transaction] = []
        for i in range(k):
            amount = float(rng.uniform(1.0, 60.0))  # micro amounts, unlike the user's norm
            merchant = ctx.pick_merchant(rng, category="ecommerce")
            t = ctx.make_txn(user, ts, amount, self.name, merchant=merchant,
                             device=device, channel="online" if "online" in ctx.config.channels else None,
                             rng=rng)
            out.append(t)
            ts = ts + timedelta(seconds=float(rng.uniform(10, 90)))
        return out


@register
class Velocity:
    name = "velocity"
    difficulty = "medium"

    def inject(self, user: User, user_txns: list[Transaction], ctx: FraudContext,
               rng: np.random.Generator, budget: int) -> list[Transaction]:
        # Many transactions within an hour at *normal* amounts: only velocity is anomalous.
        k = _clamp(budget, int(rng.integers(5, 13)))
        base = ctx.typical_amount(user_txns, user)
        device = ctx.existing_device(user, rng)
        ts = anchor_time(user_txns, rng) + minutes(rng, 1, 30)
        out: list[Transaction] = []
        for _ in range(k):
            amount = base * float(rng.uniform(0.6, 1.5))
            merchant = ctx.pick_merchant(rng)
            out.append(ctx.make_txn(user, ts, amount, self.name, merchant=merchant,
                                    device=device, rng=rng))
            ts = ts + minutes(rng, 1, 6)
        return out


@register
class MerchantFraud:
    name = "merchant"
    difficulty = "medium"

    def inject(self, user: User, user_txns: list[Transaction], ctx: FraudContext,
               rng: np.random.Generator, budget: int) -> list[Transaction]:
        # Normal amounts/devices, but at high-risk merchants — only merchant risk hints.
        k = _clamp(budget, int(rng.integers(2, 6)))
        base = ctx.typical_amount(user_txns, user)
        device = ctx.existing_device(user, rng)
        out: list[Transaction] = []
        ts = anchor_time(user_txns, rng) + minutes(rng, 5, 600)
        for _ in range(k):
            amount = base * float(rng.uniform(0.8, 2.5))
            merchant = ctx.pick_merchant(rng, high_risk=True)
            out.append(ctx.make_txn(user, ts, amount, self.name, merchant=merchant,
                                    device=device, rng=rng))
            ts = ts + minutes(rng, 60, 1440)
        return out


@register
class BeneficiaryFraud:
    name = "beneficiary"
    difficulty = "medium"

    def inject(self, user: User, user_txns: list[Transaction], ctx: FraudContext,
               rng: np.random.Generator, budget: int) -> list[Transaction]:
        # Transfers to a newly added, untrusted payee; elevated but not absurd amounts.
        k = _clamp(budget, int(rng.integers(1, 5)))
        base = ctx.typical_amount(user_txns, user)
        device = ctx.existing_device(user, rng)
        anchor = anchor_time(user_txns, rng)
        new_payee = ctx.new_beneficiary(user, ctx.day_offset(anchor), rng)
        transfer_merchant = ctx.pick_merchant(rng, category="money_transfer")
        out: list[Transaction] = []
        ts = anchor + minutes(rng, 10, 240)
        for _ in range(k):
            amount = base * float(rng.uniform(3.0, 9.0))
            # Encode the payee identity in the merchant_name so the transfer target is visible.
            t = ctx.make_txn(user, ts, amount, self.name, merchant=transfer_merchant,
                             device=device, payment_method=self._transfer_method(ctx),
                             rng=rng)
            t.merchant_name = f"Transfer to {new_payee.name}"
            out.append(t)
            ts = ts + minutes(rng, 30, 720)
        return out

    @staticmethod
    def _transfer_method(ctx: FraudContext) -> str:
        for pref in ("imps", "neft", "upi"):
            if pref in ctx.config.payment_methods:
                return pref
        return ctx.config.payment_methods[0]
