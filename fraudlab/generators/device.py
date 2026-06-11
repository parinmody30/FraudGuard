"""Device generator: per-user device pools that evolve over the timeline."""

from __future__ import annotations

import numpy as np

from fraudlab.behaviors.temporal import timeline_days
from fraudlab.config import reference_data as rd
from fraudlab.config.config import FraudLabConfig
from fraudlab.generators.rng import per_index_rng
from fraudlab.models import Device, User


class DeviceGenerator:
    """Generate a small pool of devices per user with realistic introduction dates."""

    def __init__(self, config: FraudLabConfig) -> None:
        self.config = config
        self._days = timeline_days(config)

    def generate(self, users: list[User]) -> dict[str, list[Device]]:
        devices: dict[str, list[Device]] = {}
        for idx, user in enumerate(users):
            rng = per_index_rng(self.config.random_seed, "device", idx)
            devices[user.user_id] = self._user_devices(user, rng)
        return devices

    def _user_devices(self, user: User, rng: np.random.Generator) -> list[Device]:
        # At least one device present from day 0; churn introduces extras over time.
        n_extra = int(rng.poisson(user.device_churn_rate * 3.0))
        n_devices = int(np.clip(1 + n_extra, 1, 4))

        out: list[Device] = []
        for j in range(n_devices):
            dtype = str(rng.choice(rd.DEVICE_TYPES))
            os = str(rng.choice(rd.DEVICE_OS[dtype]))
            browser = str(rng.choice(rd.DEVICE_BROWSER[dtype]))
            first_seen = 0 if j == 0 else int(rng.integers(1, self._days))
            # Trust grows with how long the device has been around within the window.
            tenure = (self._days - first_seen) / max(1, self._days)
            trust = float(np.clip(0.3 + 0.6 * tenure + rng.normal(0, 0.05), 0.05, 0.99))
            fingerprint = f"{user.user_id}-{j}-{int(rng.integers(0, 2**31)):08x}"
            out.append(
                Device(
                    device_id=f"D{user.user_id[1:]}{j:02d}",
                    device_type=dtype,
                    operating_system=os,
                    browser=browser,
                    fingerprint=fingerprint,
                    trust_score=round(trust, 4),
                    owner_user_id=user.user_id,
                    first_seen_day=first_seen,
                )
            )
        return out
