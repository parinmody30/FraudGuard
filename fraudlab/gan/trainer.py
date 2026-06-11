"""CTGAN trainer: deterministic training and artifact persistence.

Wraps the ``ctgan`` package's :class:`CTGAN` (the model SDV's ``CTGANSynthesizer`` wraps)
with an explicit, serializable configuration and reproducible-given-a-seed training.

Determinism: ``CTGAN.set_random_state(seed)`` swaps the global numpy + torch RNG state for
the duration of ``fit``/``sample`` (including network weight initialization), so two trainings
with the same config, seed, and data are identical. We also seed the global generators up
front as belt-and-suspenders.

Artifacts persisted to ``models/``:
* ``ctgan.pkl``       - the trained model (via ``CTGAN.save``).
* ``ctgan_metadata.json`` - column typing/ordering (:class:`GANMetadata`).
* ``ctgan_config.json``   - the training configuration.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from fraudlab.gan.dataset_builder import to_model_frame
from fraudlab.gan.metadata import GANMetadata

DEFAULT_MODEL_PATH = os.path.join("models", "ctgan.pkl")
DEFAULT_METADATA_PATH = os.path.join("models", "ctgan_metadata.json")
DEFAULT_CONFIG_PATH = os.path.join("models", "ctgan_config.json")


@dataclass(slots=True)
class CTGANConfig:
    """Serializable CTGAN hyperparameters.

    ``batch_size`` must be a positive multiple of ``pac`` (a CTGAN constraint); this is
    validated and a helpful error is raised otherwise.
    """

    epochs: int = 300
    batch_size: int = 500
    embedding_dim: int = 128
    generator_dim: tuple[int, ...] = (256, 256)
    discriminator_dim: tuple[int, ...] = (256, 256)
    pac: int = 10
    random_seed: int = 42
    log_frequency: bool = True
    verbose: bool = False

    def __post_init__(self) -> None:
        self.generator_dim = tuple(self.generator_dim)
        self.discriminator_dim = tuple(self.discriminator_dim)
        if self.batch_size <= 0 or self.pac <= 0:
            raise ValueError("batch_size and pac must be positive")
        if self.batch_size % self.pac != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be a multiple of pac ({self.pac})"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CTGANConfig":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in fields})

    def save(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "CTGANConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(config: CTGANConfig):
    """Instantiate a (seeded, untrained) CTGAN from a config. Imports ctgan lazily."""
    from ctgan import CTGAN

    model = CTGAN(
        embedding_dim=config.embedding_dim,
        generator_dim=config.generator_dim,
        discriminator_dim=config.discriminator_dim,
        batch_size=config.batch_size,
        pac=config.pac,
        epochs=config.epochs,
        log_frequency=config.log_frequency,
        verbose=config.verbose,
        enable_gpu=False,
    )
    model.set_random_state(config.random_seed)
    return model


def train_ctgan(
    training_frame: pd.DataFrame, metadata: GANMetadata, config: CTGANConfig
):
    """Train a CTGAN on a (soft-only, id-dropped) training frame. Returns the fitted model.

    The frame is encoded to the model representation via :func:`to_model_frame`, then fit
    with the metadata's discrete columns declared explicitly.
    """
    _seed_everything(config.random_seed)
    model = build_model(config)
    model_frame = to_model_frame(training_frame, metadata)
    model.fit(model_frame, discrete_columns=metadata.discrete_columns())
    return model


def save_artifacts(
    model,
    metadata: GANMetadata,
    config: CTGANConfig,
    *,
    model_path: str = DEFAULT_MODEL_PATH,
    metadata_path: str = DEFAULT_METADATA_PATH,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> dict[str, str]:
    """Persist model + metadata + config; return the written paths."""
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    model.save(model_path)
    metadata.save(metadata_path)
    config.save(config_path)
    return {"model": model_path, "metadata": metadata_path, "config": config_path}


def load_artifacts(
    *,
    model_path: str = DEFAULT_MODEL_PATH,
    metadata_path: str = DEFAULT_METADATA_PATH,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> tuple[object, GANMetadata, CTGANConfig | None]:
    """Load a persisted model + metadata (+ config if present)."""
    from ctgan import CTGAN

    model = CTGAN.load(model_path)
    metadata = GANMetadata.load(metadata_path)
    config = CTGANConfig.load(config_path) if os.path.exists(config_path) else None
    return model, metadata, config
