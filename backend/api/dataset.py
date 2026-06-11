"""Dataset-generation route — simulator (Phase 7.0).

Delegates to ``fraudlab.pipeline.build_dataset`` (Phase 1). Returns the run summary plus a
small preview sample for the Dataset Studio page. No simulator changes.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.api.schemas import DatasetConfigIn, DatasetSummaryOut

router = APIRouter(tags=["dataset"])


@router.post(
    "/api/dataset/generate",
    response_model=DatasetSummaryOut,
    responses={422: {"description": "Invalid configuration"}},
)
def generate_dataset(config: DatasetConfigIn) -> DatasetSummaryOut:
    """Generate a synthetic dataset and return its summary + a preview sample."""
    from fraudlab.config.config import BusinessType, FraudLabConfig, RiskAppetite
    from fraudlab.exporters.export import transactions_to_dataframe
    from fraudlab.pipeline import build_dataset

    try:
        cfg = FraudLabConfig(
            num_users=config.num_users,
            transactions_per_user=config.transactions_per_user,
            fraud_rate=config.fraud_rate,
            business_type=BusinessType(config.business_type),
            risk_appetite=RiskAppetite(config.risk_appetite),
            random_seed=config.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    transactions, summary = build_dataset(cfg)
    sample = []
    if config.preview_limit:
        df = transactions_to_dataframe(transactions).head(config.preview_limit)
        df = df.assign(timestamp=df["timestamp"].astype(str))
        sample = df.to_dict(orient="records")

    return DatasetSummaryOut(
        total_transactions=summary["total_transactions"],
        fraud_count=summary["fraud_count"],
        fraud_percentage=summary["fraud_percentage"],
        fraud_type_distribution=summary["fraud_type_distribution"],
        business_type=summary["business_type"],
        num_users=summary["num_users"],
        transactions_per_user=summary["transactions_per_user"],
        random_seed=summary["random_seed"],
        sample=sample,
    )
