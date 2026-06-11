"""Explainability routes — SHAP explainer (Phase 7.0).

Delegates to ``ChampionExplainer.frontend_response`` (Phase 5). No new logic.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_explainer, transactions_to_frame
from backend.api.schemas import ExplainRequest, ExplainResponse, ExplanationOut

router = APIRouter(tags=["explainability"])


@router.post(
    "/api/explain",
    response_model=ExplainResponse,
    responses={422: {"description": "Invalid transactions"}, 503: {"description": "Explainer unavailable"}},
)
def explain(request: ExplainRequest, explainer=Depends(get_explainer)) -> ExplainResponse:
    """Explain transactions -> prediction + SHAP contributors (waterfall-ready)."""
    df = transactions_to_frame(request.transactions)
    records = explainer.frontend_response(df)
    return ExplainResponse(explanations=[ExplanationOut(**r) for r in records])
