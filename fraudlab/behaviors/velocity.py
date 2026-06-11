"""Causal rolling features: velocity windows and expanding spend statistics.

All features are computed as-of each transaction using only prior transactions of the
same user, so they contain no future leakage and are realistic inputs for a detector.
"""

from __future__ import annotations

from datetime import timedelta

from fraudlab.models import Transaction

_ONE_HOUR = timedelta(hours=1)
_ONE_DAY = timedelta(hours=24)


class RollingFeatures:
    """Compute per-user causal rolling features in place over a transaction list."""

    @staticmethod
    def apply(user_txns: list[Transaction]) -> None:
        """Fill velocity_1h, velocity_24h, avg_user_spend, median_user_spend.

        ``user_txns`` must all belong to a single user. They are sorted by timestamp
        first. Velocity counts include the current transaction (i.e. how many of this
        user's transactions fall in the trailing window ending at this one). Spend
        stats are expanding over strictly-prior transactions.
        """
        user_txns.sort(key=lambda t: t.timestamp)
        n = len(user_txns)
        amounts: list[float] = []
        lo_1h = 0
        lo_24h = 0
        running_sum = 0.0
        for i in range(n):
            t = user_txns[i]
            ts = t.timestamp

            # Advance lower bounds for the trailing windows (two-pointer).
            while user_txns[lo_1h].timestamp < ts - _ONE_HOUR:
                lo_1h += 1
            while user_txns[lo_24h].timestamp < ts - _ONE_DAY:
                lo_24h += 1
            t.velocity_1h = i - lo_1h + 1
            t.velocity_24h = i - lo_24h + 1

            # Expanding spend stats over strictly-prior transactions.
            if amounts:
                t.avg_user_spend = round(running_sum / len(amounts), 2)
                t.median_user_spend = round(_median(amounts), 2)
            else:
                # No history yet: seed with the current amount.
                t.avg_user_spend = round(t.amount, 2)
                t.median_user_spend = round(t.amount, 2)

            amounts.append(t.amount)
            running_sum += t.amount


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0
