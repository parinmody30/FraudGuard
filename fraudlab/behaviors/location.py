"""Location behavior: home location plus occasional legitimate travel episodes."""

from __future__ import annotations

import numpy as np

from fraudlab.config import reference_data as rd
from fraudlab.models import User


def plan_travel_days(
    user: User,
    timeline_days: int,
    rng: np.random.Generator,
) -> dict[int, tuple[str, str, str]]:
    """Return a mapping day_offset -> (city, state, country) for travel episodes.

    Travel comes in multi-day episodes (a trip), governed by ``travel_frequency``. A
    minority of trips are international. Days not present in the map are spent at home.
    Because legitimate travel produces new cities (and sometimes foreign countries),
    geography alone cannot separate fraud from genuine behavior.
    """
    travel_map: dict[int, tuple[str, str, str]] = {}
    if user.travel_frequency <= 0:
        return travel_map

    # Expected number of trips over the timeline.
    expected_trips = user.travel_frequency * (timeline_days / 60.0)
    n_trips = int(rng.poisson(max(0.0, expected_trips)))
    for _ in range(n_trips):
        start_day = int(rng.integers(0, timeline_days))
        duration = int(np.clip(rng.poisson(4) + 1, 1, 14))
        if rng.random() < 0.25:
            city, country = rd.FOREIGN_CITIES[int(rng.integers(len(rd.FOREIGN_CITIES)))]
            state = "-"
        else:
            city, state = rd.INDIA_CITIES[int(rng.integers(len(rd.INDIA_CITIES)))]
            country = rd.HOME_COUNTRY
        for d in range(start_day, min(start_day + duration, timeline_days)):
            travel_map[d] = (city, state, country)
    return travel_map


def sample_location(
    user: User,
    day_offset: int,
    travel_map: dict[int, tuple[str, str, str]],
) -> tuple[str, str, str]:
    """Return (city, state, country) for a transaction on ``day_offset``."""
    if day_offset in travel_map:
        return travel_map[day_offset]
    return (user.city, user.state, user.country)
