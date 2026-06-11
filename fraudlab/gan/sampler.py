"""Sampling interface over a trained CTGAN.

Provides three levels of control:

* :meth:`generate` - unconditional sampling of ``n`` rows.
* :meth:`conditional_generate` - sample ``n`` rows of a specific ``fraud_type`` (or
  legitimate rows). The conditioned label is *enforced* on the output, since CTGAN's
  conditional vector biases but does not strictly guarantee the discrete value.
* :meth:`sample_with_prevalence` - sample a mixed batch that hits a requested fraud rate
  and per-(soft)-scenario mix.

All sampling is decoded back to the schema-typed representation and is reproducible given a
seed (the sampler re-seeds the model's RNG before each call when a seed is supplied).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudlab.gan.dataset_builder import (
    LEGIT_SENTINEL,
    SOFT_FRAUD_TYPES,
    from_model_frame,
)
from fraudlab.gan.metadata import GANMetadata


def _allocate(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Largest-remainder apportionment of ``total`` across positive-weighted keys."""
    names = [k for k, w in weights.items() if w > 0]
    if not names or total <= 0:
        return {}
    s = sum(weights[n] for n in names)
    raw = {n: total * weights[n] / s for n in names}
    floors = {n: int(np.floor(v)) for n, v in raw.items()}
    remainder = total - sum(floors.values())
    order = sorted(names, key=lambda n: (raw[n] - floors[n], n), reverse=True)
    for i in range(remainder):
        floors[order[i % len(order)]] += 1
    return floors


class CTGANSampler:
    """Decode-aware sampler wrapping a trained CTGAN + its metadata."""

    def __init__(self, model, metadata: GANMetadata, *, seed: int | None = None) -> None:
        self.model = model
        self.metadata = metadata
        self._seed = seed
        if seed is not None:
            self.model.set_random_state(seed)

    # --- low-level -------------------------------------------------------------

    def _empty(self) -> pd.DataFrame:
        return from_model_frame(
            pd.DataFrame(columns=self.metadata.columns), self.metadata
        )

    def _reseed(self, seed: int | None) -> None:
        if seed is not None:
            self.model.set_random_state(seed)

    # --- public API ------------------------------------------------------------

    def generate(self, n: int, *, seed: int | None = None) -> pd.DataFrame:
        """Unconditionally sample ``n`` decoded rows."""
        if n <= 0:
            return self._empty()
        self._reseed(seed)
        raw = self.model.sample(n)
        return from_model_frame(raw, self.metadata).reset_index(drop=True)

    def conditional_generate(
        self, fraud_type: str | None, n: int, *, seed: int | None = None
    ) -> pd.DataFrame:
        """Sample ``n`` rows conditioned on ``fraud_type`` (``None`` -> legitimate rows).

        The requested label is enforced on the decoded output so callers get exactly the
        class they asked for.
        """
        if n <= 0:
            return self._empty()
        cond_value = LEGIT_SENTINEL if fraud_type is None else fraud_type
        self._reseed(seed)
        raw = self.model.sample(n, condition_column="fraud_type", condition_value=cond_value)
        out = from_model_frame(raw, self.metadata).reset_index(drop=True)
        # Enforce the requested label (conditioning is soft).
        if fraud_type is None:
            out["fraud_type"] = None
            out["is_fraud"] = False
        else:
            out["fraud_type"] = fraud_type
            out["is_fraud"] = True
        return out

    def sample_with_prevalence(
        self,
        n: int,
        *,
        fraud_rate: float,
        fraud_mix: dict[str, float] | None = None,
        seed: int | None = None,
        shuffle: bool = True,
    ) -> pd.DataFrame:
        """Sample ``n`` rows hitting ``fraud_rate`` with a soft-only per-scenario ``fraud_mix``.

        Only easy/medium scenarios are sampleable from CTGAN; any hard-tier weights in
        ``fraud_mix`` are dropped here (the merger supplies hard fraud from the simulator).
        """
        if n <= 0:
            return self._empty()
        base_seed = self._seed if seed is None else seed

        n_fraud = int(round(n * fraud_rate))
        n_fraud = max(0, min(n, n_fraud))
        n_legit = n - n_fraud

        if fraud_mix is None:
            soft_weights = {t: 1.0 for t in SOFT_FRAUD_TYPES}
        else:
            soft_weights = {t: w for t, w in fraud_mix.items() if t in SOFT_FRAUD_TYPES and w > 0}
            if not soft_weights:
                soft_weights = {t: 1.0 for t in SOFT_FRAUD_TYPES}

        per_type = _allocate(n_fraud, soft_weights)

        parts: list[pd.DataFrame] = []
        # Distinct sub-seeds keep each conditional draw reproducible yet decorrelated.
        if n_legit > 0:
            parts.append(
                self.conditional_generate(
                    None, n_legit, seed=None if base_seed is None else base_seed + 1
                )
            )
        for i, (ftype, count) in enumerate(sorted(per_type.items())):
            if count <= 0:
                continue
            parts.append(
                self.conditional_generate(
                    ftype, count, seed=None if base_seed is None else base_seed + 100 + i
                )
            )

        if not parts:
            return self._empty()
        df = pd.concat(parts, ignore_index=True)
        if shuffle:
            rng = np.random.default_rng(0 if base_seed is None else base_seed)
            df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
        return df
