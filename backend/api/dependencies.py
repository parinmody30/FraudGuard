"""Shared FastAPI dependencies: cached service singletons + helpers (Phase 7.0).

Services (predictor / explainer / agent) are heavy to construct, so they are loaded once and
reused. Missing artifacts surface as HTTP 503 rather than crashing the app. No business logic
lives here — only wiring to existing modules.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import HTTPException

API_VERSION = "0.1.0"


def transactions_to_frame(transactions: list[Any]) -> pd.DataFrame:
    """Convert request transaction models into the DataFrame the services expect."""
    records = [t.model_dump() for t in transactions]
    df = pd.DataFrame.from_records(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


@lru_cache(maxsize=1)
def get_predictor():
    """Cached ChampionPredictor (Phase 4)."""
    try:
        from fraudlab.models.predictor import ChampionPredictor

        return ChampionPredictor.load()
    except FileNotFoundError as exc:  # pragma: no cover - depends on artifact presence
        raise HTTPException(status_code=503, detail=f"champion model unavailable: {exc}") from exc


@lru_cache(maxsize=1)
def get_explainer():
    """Cached ChampionExplainer (Phase 5)."""
    try:
        from fraudlab.explainability.shap_explainer import ChampionExplainer

        return ChampionExplainer.load()
    except FileNotFoundError as exc:  # pragma: no cover
        raise HTTPException(status_code=503, detail=f"explainer unavailable: {exc}") from exc


def get_agent(use_groq: bool = True):
    """InvestigationAgent (Phase 6). Not cached: respects the per-request groq toggle."""
    from fraudlab.agent.investigator import InvestigationAgent

    return InvestigationAgent.load(use_groq=use_groq)
