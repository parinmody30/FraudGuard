"""Beneficiary generator: recurring payees for bank/UPI users."""

from __future__ import annotations

import numpy as np

from fraudlab.behaviors.temporal import timeline_days
from fraudlab.config import reference_data as rd
from fraudlab.config.config import FraudLabConfig
from fraudlab.generators.rng import per_index_rng
from fraudlab.models import Beneficiary, User


class BeneficiaryGenerator:
    """Generate a small set of trusted, recurring beneficiaries per user."""

    def __init__(self, config: FraudLabConfig) -> None:
        self.config = config
        self._days = timeline_days(config)

    def generate(self, users: list[User]) -> dict[str, list[Beneficiary]]:
        out: dict[str, list[Beneficiary]] = {}
        for idx, user in enumerate(users):
            rng = per_index_rng(self.config.random_seed, "beneficiary", idx)
            out[user.user_id] = self._user_beneficiaries(user, idx, rng)
        return out

    def _user_beneficiaries(
        self, user: User, idx: int, rng: np.random.Generator
    ) -> list[Beneficiary]:
        n = int(np.clip(rng.poisson(3) + 1, 1, 8))
        beneficiaries: list[Beneficiary] = []
        for j in range(n):
            name = f"{rng.choice(rd.FIRST_NAMES)} {rng.choice(rd.LAST_NAMES)}"
            # Most established payees are trusted; the newest ones may not be.
            first_seen = 0 if j < n - 1 else int(rng.integers(0, max(1, self._days)))
            is_trusted = first_seen < self._days * 0.5
            beneficiaries.append(
                Beneficiary(
                    beneficiary_id=f"B{idx:06d}{j:02d}",
                    owner_user_id=user.user_id,
                    name=name,
                    account_ref=f"AC{int(rng.integers(0, 10**10)):010d}",
                    is_trusted=is_trusted,
                    first_seen_day=first_seen,
                )
            )
        return beneficiaries
