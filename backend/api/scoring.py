"""Scoring routes — champion predictor (Phase 7.0).

Delegates to ``ChampionPredictor.predict_response`` (Phase 4). No new logic.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.api.dependencies import get_predictor, transactions_to_frame
from backend.api.schemas import PredictionOut, ScoreRequest, ScoreResponse

router = APIRouter(tags=["scoring"])


@router.post(
    "/api/score",
    response_model=ScoreResponse,
    responses={422: {"description": "Invalid transactions"}, 503: {"description": "Model unavailable"}},
)
def score(request: ScoreRequest, predictor=Depends(get_predictor)) -> ScoreResponse:
    """Score transactions -> fraud probability + ALLOW/REVIEW/BLOCK + confidence."""
    df = transactions_to_frame(request.transactions)
    records = predictor.predict_response(df)
    return ScoreResponse(predictions=[PredictionOut(**r) for r in records])
