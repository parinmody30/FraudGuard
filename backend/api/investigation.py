"""Investigation routes — investigation agent (Phase 7.0).

``/api/investigate`` runs the agent over pre-computed explanations; ``/api/transactions/analyze``
is the end-to-end convenience pipeline (score -> explain -> investigate). Both delegate to
existing Phase 4/5/6 services; no new logic.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_explainer, get_predictor, transactions_to_frame
from backend.api.schemas import (
    AnalyzedTransaction,
    AnalyzeRequest,
    AnalyzeResponse,
    ExplanationOut,
    InvestigateRequest,
    InvestigateResponse,
    InvestigationOut,
    PredictionOut,
)

router = APIRouter(tags=["investigation"])


@router.post(
    "/api/investigate",
    response_model=InvestigateResponse,
    responses={422: {"description": "Invalid explanation"}},
)
def investigate(request: InvestigateRequest) -> InvestigateResponse:
    """Generate grounded investigation narratives for explanation objects."""
    from backend.api.dependencies import get_agent

    agent = get_agent(use_groq=request.use_groq)
    explanations = [e.model_dump() for e in request.explanations]
    reports = agent.investigate_batch(explanations)
    mode = "groq" if agent.groq_enabled else "fallback"
    return InvestigateResponse(
        investigations=[InvestigationOut(**r) for r in reports], mode=mode
    )


@router.post(
    "/api/transactions/analyze",
    response_model=AnalyzeResponse,
    responses={503: {"description": "Model/explainer unavailable"}},
)
def analyze(
    request: AnalyzeRequest,
    predictor=Depends(get_predictor),
    explainer=Depends(get_explainer),
) -> AnalyzeResponse:
    """End-to-end: score + explain + investigate raw transactions in one call."""
    from backend.api.dependencies import get_agent

    df = transactions_to_frame(request.transactions)
    predictions = {r["transaction_id"]: r for r in predictor.predict_response(df)}
    explanations = explainer.frontend_response(df)

    agent = get_agent(use_groq=request.use_groq)
    investigations = {r["transaction_id"]: r for r in agent.investigate_batch(explanations)}
    mode = "groq" if agent.groq_enabled else "fallback"

    results = []
    for exp in explanations:
        tid = exp["transaction_id"]
        results.append(
            AnalyzedTransaction(
                transaction_id=tid,
                prediction=PredictionOut(**predictions[tid]),
                explanation=ExplanationOut(**exp),
                investigation=InvestigationOut(**investigations[tid]),
            )
        )
    return AnalyzeResponse(results=results, mode=mode)
