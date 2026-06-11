"""Fraud engine: allocate a fraud budget across scenarios and inject it.

Every fraud transaction is a *net addition*, so the realized fraud rate equals the
configured rate up to integer rounding:

    F = round(B * r / (1 - r))   ->   F / (B + F) ≈ r

where ``B`` is the baseline (legitimate) transaction count and ``r`` the configured rate.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from fraudlab.behaviors.velocity import RollingFeatures
from fraudlab.config.config import ALL_FRAUD_TYPES, FraudLabConfig
from fraudlab.fraud.base import REGISTRY, FraudContext
from fraudlab.generators.ids import IdMinter
from fraudlab.generators.rng import stream_rng
from fraudlab.models import Beneficiary, Device, Merchant, Transaction, User


class FraudEngine:
    """Inject fraud into a baseline transaction set according to the config."""

    def __init__(
        self,
        config: FraudLabConfig,
        users: list[User],
        merchants: list[Merchant],
        devices: dict[str, list[Device]],
        beneficiaries: dict[str, list[Beneficiary]],
        id_minter: IdMinter,
    ) -> None:
        self.config = config
        self.users = users
        self.user_by_id = {u.user_id: u for u in users}
        self.ctx = FraudContext(
            config=config,
            users=self.user_by_id,
            merchants=merchants,
            devices=devices,
            beneficiaries=beneficiaries,
            id_minter=id_minter,
        )

    def inject(self, baseline_txns: list[Transaction]) -> list[Transaction]:
        baseline_count = len(baseline_txns)
        target = self._target_fraud_count(baseline_count)
        if target <= 0:
            return baseline_txns

        rng = stream_rng(self.config.random_seed, "fraud")
        allocation = self._allocate(target)

        # Group baseline transactions per user (references; we append fraud rows).
        user_map: dict[str, list[Transaction]] = defaultdict(list)
        for t in baseline_txns:
            user_map[t.user_id].append(t)

        victim_order = self._victim_cycle(rng)
        affected: set[str] = set()

        for name in ALL_FRAUD_TYPES:
            remaining = allocation.get(name, 0)
            if remaining <= 0:
                continue
            scenario = REGISTRY[name]
            while remaining > 0:
                uid = next(victim_order)
                user = self.user_by_id[uid]
                new_txns = scenario.inject(user, user_map[uid], self.ctx, rng, remaining)
                if not new_txns:
                    continue
                if len(new_txns) > remaining:
                    new_txns = new_txns[:remaining]
                user_map[uid].extend(new_txns)
                affected.add(uid)
                remaining -= len(new_txns)

        # Recompute causal rolling features for users whose timelines changed.
        for uid in affected:
            RollingFeatures.apply(user_map[uid])

        # Flatten back to a single list (sorted globally by timestamp for usability).
        merged: list[Transaction] = [t for txns in user_map.values() for t in txns]
        merged.sort(key=lambda t: (t.timestamp, t.user_id))
        return merged

    # ------------------------------------------------------------------ helpers

    def _target_fraud_count(self, baseline_count: int) -> int:
        r = self.config.fraud_rate
        if r <= 0:
            return 0
        if r >= 1:
            return baseline_count * 10  # degenerate; effectively all-fraud
        return round(baseline_count * r / (1.0 - r))

    def _allocate(self, target: int) -> dict[str, int]:
        """Largest-remainder apportionment of ``target`` across the fraud mix."""
        mix = self.config.fraud_mix or {}
        names = [n for n in ALL_FRAUD_TYPES if mix.get(n, 0.0) > 0.0]
        if not names:
            return {}
        raw = {n: target * mix[n] for n in names}
        floors = {n: int(np.floor(v)) for n, v in raw.items()}
        used = sum(floors.values())
        remainder = target - used
        # Distribute the remaining units to the largest fractional parts (stable order).
        fracs = sorted(names, key=lambda n: (raw[n] - floors[n], n), reverse=True)
        for i in range(remainder):
            floors[fracs[i % len(fracs)]] += 1
        return floors

    def _victim_cycle(self, rng: np.random.Generator):
        """Infinite deterministic generator of user ids, reshuffled each full pass."""
        ids = [u.user_id for u in self.users]
        while True:
            order = list(ids)
            rng.shuffle(order)
            yield from order
