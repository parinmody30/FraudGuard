"""Temporal behavior: timeline window, time-of-day, weekday/weekend, seasonality."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from fraudlab.config import reference_data as rd
from fraudlab.config.config import FraudLabConfig
from fraudlab.models import User


def timeline_days(config: FraudLabConfig) -> int:
    """Number of days the dataset spans.

    Scaled with transactions_per_user so per-user activity density stays realistic
    (roughly a handful of transactions per week) regardless of dataset size.
    """
    return max(60, int(config.transactions_per_user * 2.5))


def _hour_weights(user: User) -> np.ndarray:
    """A 24-length probability vector peaking on the user's active hours."""
    w = np.full(24, 0.02)
    for h in user.typical_active_hours:
        w[h % 24] += 1.0
    return w / w.sum()


def sample_timestamp(
    user: User,
    rng: np.random.Generator,
    config: FraudLabConfig,
    *,
    n: int,
) -> list[datetime]:
    """Sample ``n`` timestamps over the timeline reflecting the user's habits.

    Combines: a per-day intensity from monthly seasonality and weekday/weekend
    weighting, and an hour-of-day drawn from the user's active-hour profile.
    """
    days = timeline_days(config)
    start = datetime.combine(config.start_date, datetime.min.time())

    # Per-day weight = seasonality(month) * weekday factor * salary-cycle factor.
    day_dates = [config.start_date + timedelta(days=d) for d in range(days)]
    day_w = np.empty(days)
    for d, dt in enumerate(day_dates):
        season = rd.MONTH_SEASONALITY[dt.month - 1]
        is_weekend = dt.weekday() >= 5
        weekend_factor = 1.25 if is_weekend else 1.0  # more discretionary spend on weekends
        # Spending clusters in the week following salary credit.
        days_since_salary = (dt.day - user.salary_day) % 30
        salary_factor = 1.4 if days_since_salary <= 6 else 1.0
        day_w[d] = season * weekend_factor * salary_factor
    day_w /= day_w.sum()

    hour_w = _hour_weights(user)

    chosen_days = rng.choice(days, size=n, p=day_w)
    timestamps: list[datetime] = []
    for d in chosen_days:
        hour = int(rng.choice(24, p=hour_w))
        minute = int(rng.integers(0, 60))
        second = int(rng.integers(0, 60))
        timestamps.append(start + timedelta(days=int(d), hours=hour, minutes=minute, seconds=second))
    timestamps.sort()
    return timestamps
