"""Persist and load the champion model bundle (Phase 4).

The bundle (fitted preprocessor + fitted XGBoost + feature names + decision policy) is saved
as a single joblib artifact; the metadata is additionally written as human-readable JSON so
it can be surfaced in the frontend dashboard without unpickling the model.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any

from fraudlab.models.champion import (
    ChampionBundle,
    ConfidenceBands,
    DecisionThresholds,
)

DEFAULT_MODEL_PATH = os.path.join("models", "champion_model.joblib")
DEFAULT_METADATA_PATH = os.path.join("models", "champion_metadata.json")


def save_bundle(
    bundle: ChampionBundle,
    *,
    model_path: str = DEFAULT_MODEL_PATH,
    metadata_path: str = DEFAULT_METADATA_PATH,
) -> dict[str, str]:
    """Save the bundle (joblib) and its metadata (JSON). Returns the written paths."""
    import joblib

    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    payload = {
        "preprocessor": bundle.preprocessor,
        "model": bundle.model,
        "feature_names": bundle.feature_names,
        "metadata": bundle.metadata,
        "thresholds": asdict(bundle.thresholds),
        "confidence": asdict(bundle.confidence),
    }
    joblib.dump(payload, model_path)

    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(bundle.metadata, fh, indent=2, default=str)

    return {"model": model_path, "metadata": metadata_path}


def load_bundle(*, model_path: str = DEFAULT_MODEL_PATH) -> ChampionBundle:
    """Load a persisted champion bundle from a joblib artifact."""
    import joblib

    payload: dict[str, Any] = joblib.load(model_path)
    return ChampionBundle(
        preprocessor=payload["preprocessor"],
        model=payload["model"],
        feature_names=list(payload["feature_names"]),
        metadata=payload["metadata"],
        thresholds=DecisionThresholds(**payload.get("thresholds", {})),
        confidence=ConfidenceBands(**payload.get("confidence", {})),
    )
