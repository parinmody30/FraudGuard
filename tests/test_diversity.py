"""Population and behavioral diversity checks (avoid homogeneous output)."""

from __future__ import annotations

import statistics
from collections import defaultdict

from fraudlab.generators import PopulationGenerator


def test_population_attributes_vary(small_config):
    users, _ = PopulationGenerator(small_config).generate()
    assert len({u.occupation for u in users}) >= 5
    assert len({u.income_band for u in users}) >= 3
    assert len({u.archetype for u in users}) >= 3
    assert len({u.city for u in users}) >= 5
    # Spend parameters should span a meaningful range.
    mus = [u.spend_mu for u in users]
    assert statistics.pstdev(mus) > 0.2


def test_per_user_spend_differs(small_dataset):
    transactions, _, _ = small_dataset
    by_user = defaultdict(list)
    for t in transactions:
        if not t.is_fraud:
            by_user[t.user_id].append(t.amount)
    means = [statistics.mean(v) for v in by_user.values() if v]
    # Users should not all have the same average spend.
    assert statistics.pstdev(means) > 1.0
    assert max(means) > 2 * min(means)


def test_multiple_cities_and_devices(small_dataset):
    transactions, _, _ = small_dataset
    assert len({t.city for t in transactions}) >= 5
    assert len({t.device_id for t in transactions}) >= 10
    # Some legitimate foreign activity exists (travel), so geography isn't a clean rule.
    assert any(t.is_foreign and not t.is_fraud for t in transactions)
