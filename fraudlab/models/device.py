"""Device model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Device:
    """A device used by a user.

    ``trust_score`` (0..1) grows with usage age; ``first_seen_day`` is the timeline day
    (offset from the dataset start) on which the device was first observed, enabling
    realistic device evolution and "new device" baselines.
    """

    device_id: str
    device_type: str
    operating_system: str
    browser: str
    fingerprint: str
    trust_score: float
    owner_user_id: str
    first_seen_day: int = 0
