"""Fraud scenario protocol, shared injection context, and the scenario registry.

Design notes
------------
* Every fraud transaction is a **net addition** to the dataset (scenarios never relabel
  a legitimate transaction). This keeps baseline labels pristine and lets the engine hit
  an exact fraud rate by simple arithmetic.
* Scenarios receive a victim ``User``, that user's current transaction list (legit rows,
  for timing/location context), a :class:`FraudContext` of shared resources, an ``rng``,
  and a ``budget`` (max fraud rows to create). They return the list of new fraud
  transactions. Rolling features are recomputed by the engine afterward, so scenarios
  leave velocity/spend fields at zero.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

import numpy as np

from fraudlab.behaviors.temporal import timeline_days
from fraudlab.config import reference_data as rd
from fraudlab.config.config import FraudLabConfig
from fraudlab.generators.ids import IdMinter
from fraudlab.models import Beneficiary, Device, Merchant, Transaction, User

HIGH_RISK_CATEGORIES = ("crypto", "gambling", "money_transfer", "wallet_topup")


@dataclass(slots=True)
class FraudContext:
    """Shared resources and helpers available to every scenario."""

    config: FraudLabConfig
    users: dict[str, User]
    merchants: list[Merchant]
    devices: dict[str, list[Device]]
    beneficiaries: dict[str, list[Beneficiary]]
    id_minter: IdMinter
    by_category: dict[str, list[Merchant]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.by_category:
            idx: dict[str, list[Merchant]] = defaultdict(list)
            for m in self.merchants:
                idx[m.merchant_category].append(m)
            self.by_category = dict(idx)

    # --- timing / geography helpers -------------------------------------------

    @property
    def days(self) -> int:
        return timeline_days(self.config)

    def day_offset(self, ts: datetime) -> int:
        return max(0, min(self.days - 1, (ts.date() - self.config.start_date).days))

    def typical_amount(self, user_txns: list[Transaction], user: User) -> float:
        amounts = [t.amount for t in user_txns if not t.is_fraud]
        if amounts:
            return float(np.median(amounts))
        return float(np.exp(user.spend_mu))

    def foreign_location(self, rng: np.random.Generator) -> tuple[str, str, str]:
        city, country = rd.FOREIGN_CITIES[int(rng.integers(len(rd.FOREIGN_CITIES)))]
        return city, "-", country

    def distant_domestic_location(
        self, user: User, rng: np.random.Generator
    ) -> tuple[str, str, str]:
        choices = [c for c in rd.INDIA_CITIES if c[0] != user.city]
        city, state = choices[int(rng.integers(len(choices)))]
        return city, state, rd.HOME_COUNTRY

    # --- merchant / device helpers --------------------------------------------

    def pick_merchant(
        self,
        rng: np.random.Generator,
        *,
        category: str | None = None,
        high_risk: bool = False,
    ) -> Merchant:
        if high_risk:
            pool = [m for m in self.merchants if m.merchant_category in HIGH_RISK_CATEGORIES]
            if pool:
                return pool[int(rng.integers(len(pool)))]
        if category is not None and self.by_category.get(category):
            pool = self.by_category[category]
            return pool[int(rng.integers(len(pool)))]
        return self.merchants[int(rng.integers(len(self.merchants)))]

    def existing_device(
        self, user: User, rng: np.random.Generator
    ) -> Device:
        pool = self.devices[user.user_id]
        # Favor the established primary device so fraud need not involve a new device.
        return pool[0] if rng.random() < 0.7 else pool[int(rng.integers(len(pool)))]

    def new_device(
        self, user: User, day_offset: int, rng: np.random.Generator
    ) -> Device:
        dtype = str(rng.choice(rd.DEVICE_TYPES))
        os = str(rng.choice(rd.DEVICE_OS[dtype]))
        browser = str(rng.choice(rd.DEVICE_BROWSER[dtype]))
        return Device(
            device_id=f"DFRD{int(rng.integers(0, 2**31)):08x}",
            device_type=dtype,
            operating_system=os,
            browser=browser,
            fingerprint=f"frd-{int(rng.integers(0, 2**31)):08x}",
            trust_score=round(float(rng.uniform(0.02, 0.15)), 4),
            owner_user_id=user.user_id,
            first_seen_day=day_offset,
        )

    def new_beneficiary(
        self, user: User, day_offset: int, rng: np.random.Generator
    ) -> Beneficiary:
        name = f"{rng.choice(rd.FIRST_NAMES)} {rng.choice(rd.LAST_NAMES)}"
        return Beneficiary(
            beneficiary_id=f"BFRD{int(rng.integers(0, 2**31)):08x}",
            owner_user_id=user.user_id,
            name=name,
            account_ref=f"AC{int(rng.integers(0, 10**10)):010d}",
            is_trusted=False,
            first_seen_day=day_offset,
        )

    # --- transaction factory ---------------------------------------------------

    def make_txn(
        self,
        user: User,
        ts: datetime,
        amount: float,
        fraud_type: str,
        *,
        merchant: Merchant,
        device: Device,
        payment_method: str | None = None,
        channel: str | None = None,
        location: tuple[str, str, str] | None = None,
        rng: np.random.Generator | None = None,
    ) -> Transaction:
        offset = self.day_offset(ts)
        if location is None:
            location = (user.city, user.state, user.country)
        city, state, country = location
        if payment_method is None:
            pm_pool = merchant.supported_payment_methods or list(self.config.payment_methods)
            payment_method = pm_pool[0] if rng is None else str(rng.choice(pm_pool))
        if channel is None:
            ch = self.config.channels
            channel = ch[0] if rng is None else str(rng.choice(ch))
        return Transaction(
            transaction_id=self.id_minter.next(),
            user_id=user.user_id,
            timestamp=ts,
            amount=round(max(1.0, amount), 2),
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
            account_age_days=user.account_age_days + offset,
            avg_user_spend=0.0,
            median_user_spend=0.0,
            velocity_1h=0,
            velocity_24h=0,
            merchant_risk_score=merchant.merchant_risk_score,
            device_risk_score=round(1.0 - device.trust_score, 4),
            is_fraud=True,
            fraud_type=fraud_type,
        )


@runtime_checkable
class FraudScenario(Protocol):
    """Protocol implemented by every fraud scenario."""

    name: str
    difficulty: str

    def inject(
        self,
        user: User,
        user_txns: list[Transaction],
        ctx: FraudContext,
        rng: np.random.Generator,
        budget: int,
    ) -> list[Transaction]:
        ...


# --- registry ----------------------------------------------------------------

REGISTRY: dict[str, FraudScenario] = {}


def register(cls: type) -> type:
    """Class decorator that instantiates a scenario and registers it by name."""
    instance = cls()
    REGISTRY[instance.name] = instance
    return cls


# --- small shared helpers for scenarios --------------------------------------

def anchor_time(user_txns: list[Transaction], rng: np.random.Generator) -> datetime:
    """Pick a realistic anchor timestamp from the victim's existing activity."""
    legit = [t for t in user_txns if not t.is_fraud]
    if legit:
        return legit[int(rng.integers(len(legit)))].timestamp
    if user_txns:
        return user_txns[0].timestamp
    # Deterministic fallback (should not happen: every user has baseline activity).
    return datetime(2024, 6, 1, 12, 0, 0)


def minutes(rng: np.random.Generator, lo: float, hi: float) -> timedelta:
    return timedelta(minutes=float(rng.uniform(lo, hi)))
