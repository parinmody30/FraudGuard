"""Hard fraud scenarios.

These are designed so that no single transaction looks anomalous: the fraud signal lives
in the *sequence* — a slow escalation, a fan-in/fan-out money flow, correlated activity
across channels, or a staged setup-then-cashout. Detecting them requires temporal or
relational context, defeating naive per-row rules.
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


def _days_jitter(rng: np.random.Generator, lo: float, hi: float) -> timedelta:
    return timedelta(days=float(rng.uniform(lo, hi)),
                     hours=float(rng.uniform(0, 12)))


@register
class SyntheticIdentity:
    name = "synthetic_identity"
    difficulty = "hard"

    def inject(self, user: User, user_txns: list[Transaction], ctx: FraudContext,
               rng: np.random.Generator, budget: int) -> list[Transaction]:
        # A fabricated identity quietly builds benign history, then busts out.
        k = _clamp(budget, int(rng.integers(5, 11)))
        base = ctx.typical_amount(user_txns, user)
        device = ctx.new_device(user, 0, rng)  # one consistent synthetic device
        ts = anchor_time(user_txns, rng)
        out: list[Transaction] = []
        for i in range(k):
            progress = i / max(1, k - 1)
            # Benign-looking early activity ramping toward a final cash-out.
            amount = base * (0.5 + progress * float(rng.uniform(4.0, 7.0)))
            is_cashout = progress > 0.7
            merchant = ctx.pick_merchant(rng, high_risk=is_cashout and rng.random() < 0.7)
            out.append(ctx.make_txn(user, ts, amount, self.name, merchant=merchant,
                                    device=device, rng=rng))
            ts = ts + _days_jitter(rng, 1, 6)
        return out


@register
class MoneyMule:
    name = "money_mule"
    difficulty = "hard"

    def inject(self, user: User, user_txns: list[Transaction], ctx: FraudContext,
               rng: np.random.Generator, budget: int) -> list[Transaction]:
        # Fan-in: several inbound credits, then fan-out to multiple new payees shortly
        # after. Each transfer is individually unremarkable in size.
        k = _clamp(budget, int(rng.integers(4, 10)))
        base = ctx.typical_amount(user_txns, user)
        device = ctx.existing_device(user, rng)
        transfer_merchant = ctx.pick_merchant(rng, category="money_transfer")
        n_in = max(1, k // 2)
        ts = anchor_time(user_txns, rng)
        out: list[Transaction] = []
        # Fan-in
        for _ in range(n_in):
            amount = base * float(rng.uniform(1.5, 4.0))
            t = ctx.make_txn(user, ts, amount, self.name, merchant=transfer_merchant,
                             device=device, payment_method=self._method(ctx), rng=rng)
            t.merchant_name = f"Inbound from {rng.choice(['Acc', 'Payer', 'Src'])}{int(rng.integers(1000, 9999))}"
            out.append(t)
            ts = ts + minutes(rng, 20, 240)
        # Fan-out shortly afterward to several new payees
        for _ in range(k - n_in):
            payee = ctx.new_beneficiary(user, ctx.day_offset(ts), rng)
            amount = base * float(rng.uniform(1.2, 3.5))
            t = ctx.make_txn(user, ts, amount, self.name, merchant=transfer_merchant,
                             device=device, payment_method=self._method(ctx), rng=rng)
            t.merchant_name = f"Transfer to {payee.name}"
            out.append(t)
            ts = ts + minutes(rng, 10, 120)
        return out

    @staticmethod
    def _method(ctx: FraudContext) -> str:
        for pref in ("imps", "upi", "neft"):
            if pref in ctx.config.payment_methods:
                return pref
        return ctx.config.payment_methods[0]


@register
class GradualEscalation:
    name = "gradual_escalation"
    difficulty = "hard"

    def inject(self, user: User, user_txns: list[Transaction], ctx: FraudContext,
               rng: np.random.Generator, budget: int) -> list[Transaction]:
        # Amounts creep up by a small factor each step over weeks; every single step is
        # plausible, only the monotone trend over time betrays it.
        k = _clamp(budget, int(rng.integers(6, 13)))
        base = ctx.typical_amount(user_txns, user)
        device = ctx.existing_device(user, rng)
        category = str(rng.choice(user.preferred_merchant_categories)) if user.preferred_merchant_categories else None
        ts = anchor_time(user_txns, rng)
        amount = base * float(rng.uniform(0.8, 1.2))
        out: list[Transaction] = []
        for _ in range(k):
            merchant = ctx.pick_merchant(rng, category=category)
            out.append(ctx.make_txn(user, ts, amount, self.name, merchant=merchant,
                                    device=device, rng=rng))
            amount *= float(rng.uniform(1.12, 1.35))
            ts = ts + _days_jitter(rng, 2, 7)
        return out


@register
class CrossChannel:
    name = "cross_channel"
    difficulty = "hard"

    def inject(self, user: User, user_txns: list[Transaction], ctx: FraudContext,
               rng: np.random.Generator, budget: int) -> list[Transaction]:
        # Correlated activity spread across distinct payment methods / channels in a
        # short period; each is normal, the coordination across rails is the signal.
        k = _clamp(budget, int(rng.integers(3, 7)))
        base = ctx.typical_amount(user_txns, user)
        device = ctx.existing_device(user, rng)
        methods = list(ctx.config.payment_methods)
        channels = list(ctx.config.channels)
        ts = anchor_time(user_txns, rng) + minutes(rng, 5, 180)
        out: list[Transaction] = []
        for i in range(k):
            pm = methods[i % len(methods)]
            ch = channels[i % len(channels)]
            amount = base * float(rng.uniform(0.9, 2.2))
            merchant = ctx.pick_merchant(rng)
            out.append(ctx.make_txn(user, ts, amount, self.name, merchant=merchant,
                                    device=device, payment_method=pm, channel=ch, rng=rng))
            ts = ts + minutes(rng, 20, 300)
        return out


@register
class MultiStage:
    name = "multi_stage"
    difficulty = "hard"

    def inject(self, user: User, user_txns: list[Transaction], ctx: FraudContext,
               rng: np.random.Generator, budget: int) -> list[Transaction]:
        # Stage 1 recon (tiny test), stage 2 setup (new device, small move), stage 3
        # cash-out (large) days later. The stages only make sense as a linked sequence.
        base = ctx.typical_amount(user_txns, user)
        normal_device = ctx.existing_device(user, rng)
        ts = anchor_time(user_txns, rng)
        new_device = ctx.new_device(user, ctx.day_offset(ts), rng)
        out: list[Transaction] = []

        # Stage 1: recon test transaction (micro amount, normal device).
        out.append(ctx.make_txn(user, ts, float(rng.uniform(1, 30)), self.name,
                                merchant=ctx.pick_merchant(rng, category="ecommerce"),
                                device=normal_device, rng=rng))
        ts = ts + _days_jitter(rng, 1, 4)

        # Stage 2: setup from a new device, modest amount.
        if len(out) < budget:
            out.append(ctx.make_txn(user, ts, base * float(rng.uniform(0.8, 1.5)), self.name,
                                    merchant=ctx.pick_merchant(rng), device=new_device, rng=rng))
            ts = ts + _days_jitter(rng, 2, 6)

        # Stage 3: cash-out, large, same new device, high-risk merchant.
        if len(out) < budget:
            out.append(ctx.make_txn(user, ts, base * float(rng.uniform(6.0, 14.0)), self.name,
                                    merchant=ctx.pick_merchant(rng, high_risk=True),
                                    device=new_device, rng=rng))

        # Optional extra cash-outs if budget allows.
        while len(out) < budget and len(out) < 6:
            ts = ts + minutes(rng, 30, 600)
            out.append(ctx.make_txn(user, ts, base * float(rng.uniform(4.0, 10.0)), self.name,
                                    merchant=ctx.pick_merchant(rng, high_risk=True),
                                    device=new_device, rng=rng))
        return out[:max(1, budget)]
