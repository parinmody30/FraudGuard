"""FraudLab API routers (Phase 7.0). Thin delegation to existing services; no new logic."""

from __future__ import annotations

from backend.api import dataset, explainability, health, investigation, scoring

ALL_ROUTERS = (
    health.router,
    scoring.router,
    explainability.router,
    investigation.router,
    dataset.router,
)

__all__ = ["ALL_ROUTERS"]
