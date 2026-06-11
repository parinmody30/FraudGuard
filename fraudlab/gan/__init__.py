"""CTGAN realism-augmentation subpackage (Phase 2).

CTGAN is used here strictly as a *per-row realism augmenter*: it learns the marginal/joint
distribution of individual legitimate, easy-fraud, and medium-fraud transactions. It does
**not** replace the simulator and is deliberately never trained on the hard (sequential)
fraud scenarios — those remain simulator-generated and are spliced back in by the merger.

This subpackage is self-contained and read-only with respect to the generator: it imports
only the public config/schema and never mutates generators/ fraud/ behaviors/ pipeline.

Public surface:
    GANMetadata, infer_metadata                  (metadata)
    build_training_frame, to_model_frame, ...     (dataset_builder)
    CTGANConfig, train_ctgan, save_artifacts, ... (trainer)
    CTGANSampler                                  (sampler)
    evaluate_ctgan                                (evaluator)
    build_hybrid_dataset                          (merger)
"""

from __future__ import annotations

from fraudlab.gan.dataset_builder import (
    LEGIT_SENTINEL,
    build_training_frame,
    from_model_frame,
    to_model_frame,
)
from fraudlab.gan.merger import build_hybrid_dataset
from fraudlab.gan.metadata import GANMetadata, infer_metadata
from fraudlab.gan.sampler import CTGANSampler
from fraudlab.gan.trainer import CTGANConfig, load_artifacts, save_artifacts, train_ctgan

__all__ = [
    "GANMetadata",
    "infer_metadata",
    "build_training_frame",
    "to_model_frame",
    "from_model_frame",
    "LEGIT_SENTINEL",
    "CTGANConfig",
    "train_ctgan",
    "save_artifacts",
    "load_artifacts",
    "CTGANSampler",
    "build_hybrid_dataset",
]
