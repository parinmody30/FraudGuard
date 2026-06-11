"""Deterministic RNG helpers.

All randomness in FraudLab flows from a single master seed. Each subsystem gets an
*independent* child generator derived from the master seed via ``numpy.random.SeedSequence``
so that adding, removing, or reordering a subsystem does not perturb another subsystem's
random stream. This keeps generation reproducible and modular.
"""

from __future__ import annotations

import numpy as np

# Stable integer offsets per named stream. Changing these changes the dataset, so they
# are fixed constants rather than derived from iteration order.
_STREAM_IDS: dict[str, int] = {
    "population": 1,
    "merchant": 2,
    "device": 3,
    "beneficiary": 4,
    "baseline": 5,
    "fraud": 6,
}


def stream_rng(seed: int, stream: str) -> np.random.Generator:
    """Return an independent Generator for a named subsystem stream."""
    if stream not in _STREAM_IDS:
        raise KeyError(f"Unknown RNG stream: {stream!r}")
    return np.random.default_rng(np.random.SeedSequence([seed, _STREAM_IDS[stream]]))


def per_index_rng(seed: int, stream: str, index: int) -> np.random.Generator:
    """Return an independent Generator for a specific item within a stream.

    Used for per-user generation so that each user's behavioral stream is independent
    and stable regardless of how many users precede it.
    """
    if stream not in _STREAM_IDS:
        raise KeyError(f"Unknown RNG stream: {stream!r}")
    return np.random.default_rng(
        np.random.SeedSequence([seed, _STREAM_IDS[stream], index])
    )
