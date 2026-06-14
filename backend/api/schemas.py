"""Pydantic request/response schemas — the typed backend/frontend contract (Phase 7.0).

These mirror the existing module outputs exactly:
* ``PredictionOut``       <- ChampionPredictor.predict_response
* ``ExplanationOut``      <- ChampionExplainer.frontend_response
* ``InvestigationOut``    <- InvestigationAgent.frontend_response
No new logic is introduced; the schemas only describe data already produced upstream.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------- transactions

class TransactionIn(BaseModel):
    """A single transaction record (canonical simulator schema).

    Label fields (``is_fraud`` / ``fraud_type``) are optional and ignored at scoring time.
    Extra keys are allowed so callers may forward full rows without breaking the contract.
    """

    model_config = ConfigDict(extra="allow")

    transaction_id: str
    user_id: str
    timestamp: str
    amount: float
    currency: str = "INR"
    merchant_id: str
    merchant_name: str
    merchant_category: str
    payment_method: str
    channel: str
    device_id: str
    device_type: str
    city: str
    state: str
    country: str
    is_foreign: bool
    account_age_days: int
    avg_user_spend: float
    median_user_spend: float
    velocity_1h: int
    velocity_24h: int
    merchant_risk_score: float
    device_risk_score: float
    is_fraud: bool | None = None
    fraud_type: str | None = None


# --------------------------------------------------------------------- scoring

class PredictionOut(BaseModel):
    transaction_id: str
    fraud_probability: float = Field(ge=0.0, le=1.0)
    decision: Literal["ALLOW", "REVIEW", "BLOCK"]
    confidence: Literal["LOW", "MEDIUM", "HIGH"]


class ScoreRequest(BaseModel):
    transactions: list[TransactionIn] = Field(min_length=1)


class ScoreResponse(BaseModel):
    predictions: list[PredictionOut]


# --------------------------------------------------------------------- explainability

class ContributorOut(BaseModel):
    feature: str
    impact: float
    direction: Literal["increase", "decrease"]


class ExplanationOut(BaseModel):
    transaction_id: str
    fraud_probability: float = Field(ge=0.0, le=1.0)
    decision: Literal["ALLOW", "REVIEW", "BLOCK"]
    confidence: Literal["LOW", "MEDIUM", "HIGH"]
    contributors: list[ContributorOut]


class ExplainRequest(BaseModel):
    transactions: list[TransactionIn] = Field(min_length=1)


class ExplainResponse(BaseModel):
    explanations: list[ExplanationOut]


# --------------------------------------------------------------------- investigation

class InvestigationOut(BaseModel):
    transaction_id: str
    summary: str
    signals_detected: list[str]
    recommendation: str
    confidence_explanation: str
    next_steps: list[str]


class InvestigateRequest(BaseModel):
    """Investigate pre-computed explanations (Phase 5 output)."""

    explanations: list[ExplanationOut] = Field(min_length=1)
    use_groq: bool = True


class InvestigateResponse(BaseModel):
    investigations: list[InvestigationOut]
    mode: Literal["groq", "fallback"]


class AnalyzeRequest(BaseModel):
    """End-to-end: score + explain + investigate raw transactions in one call."""

    transactions: list[TransactionIn] = Field(min_length=1)
    use_groq: bool = True


class AnalyzedTransaction(BaseModel):
    transaction_id: str
    prediction: PredictionOut
    explanation: ExplanationOut
    investigation: InvestigationOut


class AnalyzeResponse(BaseModel):
    results: list[AnalyzedTransaction]
    mode: Literal["groq", "fallback"]


# --------------------------------------------------------------------- dataset

class DatasetConfigIn(BaseModel):
    num_users: int = Field(default=200, ge=1, le=5000)
    transactions_per_user: int = Field(default=50, ge=1, le=500)
    fraud_rate: float = Field(default=0.06, ge=0.0, le=1.0)
    business_type: Literal["bank", "credit_card", "upi", "ecommerce"] = "bank"
    risk_appetite: Literal["low", "medium", "high"] = "medium"
    seed: int = 42
    preview_limit: int = Field(default=20, ge=0, le=200)


class DatasetSummaryOut(BaseModel):
    total_transactions: int
    fraud_count: int
    fraud_percentage: float
    fraud_type_distribution: dict[str, int]
    business_type: str
    num_users: int
    transactions_per_user: int
    random_seed: int
    sample: list[dict[str, Any]]
    # Persisted-run identifier (None when the database is unavailable).
    run_id: str | None = None


class DatasetRunListItem(BaseModel):
    """A persisted dataset generation, summary fields only (no transaction rows)."""

    id: str
    created_at: str
    label: str
    business_type: str
    risk_appetite: str
    num_users: int
    transactions_per_user: int
    fraud_rate: float
    seed: int
    total_transactions: int
    fraud_count: int
    fraud_percentage: float
    stored_count: int


class DatasetRunDetailOut(DatasetRunListItem):
    """A persisted run including its capped sample of transaction rows."""

    fraud_type_distribution: dict[str, int]
    transactions: list[dict[str, Any]]


# --------------------------------------------------------------------- model / health

class ModelMetadataOut(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_type: str | None = None
    champion: str | None = None
    training_timestamp: str | None = None
    dataset_size: int | None = None
    fraud_rate: float | None = None
    metrics: dict[str, Any] | None = None
    thresholds: dict[str, Any] | None = None
    confidence_bands: dict[str, Any] | None = None


class HealthOut(BaseModel):
    status: Literal["ok"]
    service: str = "fraudlab-api"
    version: str


class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None
