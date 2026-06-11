"""Merchant generator: a diverse pool of merchants with category-correlated risk."""

from __future__ import annotations

import numpy as np

from fraudlab.config import reference_data as rd
from fraudlab.config.config import FraudLabConfig
from fraudlab.generators.rng import stream_rng
from fraudlab.models import Merchant


class MerchantGenerator:
    """Generate a shared pool of merchants users can transact with."""

    def __init__(self, config: FraudLabConfig) -> None:
        self.config = config

    def pool_size(self) -> int:
        return max(50, self.config.num_users // 2)

    def generate(self) -> list[Merchant]:
        rng = stream_rng(self.config.random_seed, "merchant")
        categories = list(rd.MERCHANT_CATEGORIES.keys())
        merchants: list[Merchant] = []
        size = self.pool_size()
        for i in range(size):
            category = str(rng.choice(categories))
            base_risk, _mult = rd.MERCHANT_CATEGORIES[category]
            # Risk correlates with category but is noisy, so it is not a clean rule.
            risk = float(np.clip(rng.normal(base_risk, 0.12), 0.01, 0.99))

            prefix = str(rng.choice(rd.MERCHANT_NAME_PREFIX[category]))
            name = f"{prefix} #{int(rng.integers(1, 9999)):04d}"

            # Most merchants are domestic; a minority are foreign-based.
            if rng.random() < 0.12:
                city, country = rd.FOREIGN_CITIES[int(rng.integers(len(rd.FOREIGN_CITIES)))]
            else:
                city, _state = rd.INDIA_CITIES[int(rng.integers(len(rd.INDIA_CITIES)))]
                country = rd.HOME_COUNTRY

            methods = self._supported_methods(rng)

            merchants.append(
                Merchant(
                    merchant_id=f"M{i:06d}",
                    merchant_name=name,
                    merchant_category=category,
                    merchant_risk_score=round(risk, 4),
                    city=city,
                    country=country,
                    supported_payment_methods=methods,
                )
            )
        return merchants

    def _supported_methods(self, rng: np.random.Generator) -> list[str]:
        available = list(self.config.payment_methods)
        if len(available) == 1:
            return available
        k = int(rng.integers(1, len(available) + 1))
        return list(rng.choice(available, size=k, replace=False))
