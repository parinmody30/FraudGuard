"""Health + model-metadata routes (Phase 7.0)."""

from __future__ import annotations

import json
import os

from fastapi import APIRouter, HTTPException

from backend.api.dependencies import API_VERSION
from backend.api.schemas import HealthOut, ModelMetadataOut

router = APIRouter(tags=["system"])

METADATA_PATH = os.path.join("models", "champion_metadata.json")


@router.get("/api/health", response_model=HealthOut)
def health() -> HealthOut:
    """Liveness probe for the frontend / infra."""
    return HealthOut(status="ok", version=API_VERSION)


@router.get(
    "/api/model/metadata",
    response_model=ModelMetadataOut,
    responses={503: {"description": "Model metadata not available"}},
)
def model_metadata() -> ModelMetadataOut:
    """Return champion model metadata (metrics, thresholds) for the dashboard."""
    if not os.path.exists(METADATA_PATH):
        raise HTTPException(status_code=503, detail="champion_metadata.json not found")
    with open(METADATA_PATH, "r", encoding="utf-8") as fh:
        return ModelMetadataOut(**json.load(fh))
