"""Dataset-generation + history routes.

``POST /api/dataset/generate`` builds a synthetic dataset via ``fraudlab.pipeline.build_dataset``
and persists the run (summary + a capped sample of transaction rows) to Postgres so it can be
replayed from the Dashboard without regenerating. History is exposed via the ``/api/dataset/history``
endpoints. When the database is unavailable, generation still succeeds (history is just disabled).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Response

from backend.api.schemas import (
    DatasetConfigIn,
    DatasetRunDetailOut,
    DatasetRunListItem,
    DatasetSummaryOut,
)

logger = logging.getLogger("fraudlab.dataset")

router = APIRouter(tags=["dataset"])

# How many transaction rows to persist per run (enough to drive dashboard charts + the
# transactions table from history; keeps the database lean for large generations).
STORE_LIMIT = 2000


@router.post(
    "/api/dataset/generate",
    response_model=DatasetSummaryOut,
    responses={422: {"description": "Invalid configuration"}},
)
def generate_dataset(config: DatasetConfigIn) -> DatasetSummaryOut:
    """Generate a synthetic dataset, persist it, and return its summary + a preview sample."""
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

    full_df = transactions_to_dataframe(transactions)
    full_df = full_df.assign(timestamp=full_df["timestamp"].astype(str))
    # Missing values (e.g. fraud_type on legitimate rows) surface as NaN, which is not valid
    # JSON — Postgres rejects it and the history insert fails. Normalize them to None/null.
    full_df = full_df.astype(object).where(full_df.notna(), None)

    # Rows are chronological, so head() would only cover the first days of the timeline
    # (when users have almost no history). Take a seeded random sample across the whole
    # timeline instead — deterministic, preserves the fraud rate — and keep it chronological.
    if len(full_df) > STORE_LIMIT:
        sampled_df = full_df.sample(n=STORE_LIMIT, random_state=config.seed).sort_values(
            "timestamp", kind="stable"
        )
    else:
        sampled_df = full_df

    sample = []
    if config.preview_limit:
        sample = sampled_df.head(config.preview_limit).to_dict(orient="records")

    # Persist run with a capped sample of rows for history replay.
    stored_rows = sampled_df.to_dict(orient="records")
    run_id = _persist_run(config, summary, stored_rows)

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
        run_id=run_id,
    )


@router.get("/api/dataset/history", response_model=list[DatasetRunListItem])
def list_history() -> list[DatasetRunListItem]:
    """Return persisted dataset runs, newest first (summary fields only)."""
    from backend.db import DB_AVAILABLE, session_scope

    if not DB_AVAILABLE:
        return []
    from backend.models_db import DatasetRun

    with session_scope() as db:
        runs = db.query(DatasetRun).order_by(DatasetRun.created_at.desc()).all()
        return [DatasetRunListItem(**run.as_list_item()) for run in runs]


@router.get("/api/dataset/history/{run_id}", response_model=DatasetRunDetailOut)
def get_history(run_id: str) -> DatasetRunDetailOut:
    """Return a single persisted run including its stored transaction rows."""
    from backend.db import DB_AVAILABLE, session_scope

    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Dataset history is unavailable (database offline).")
    from backend.models_db import DatasetRun

    with session_scope() as db:
        run = db.get(DatasetRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Dataset run not found.")
        return DatasetRunDetailOut(**run.as_detail())


@router.delete("/api/dataset/history/{run_id}", status_code=204, response_class=Response)
def delete_history(run_id: str) -> Response:
    """Delete a persisted run."""
    from backend.db import DB_AVAILABLE, session_scope

    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Dataset history is unavailable (database offline).")
    from backend.models_db import DatasetRun

    with session_scope() as db:
        run = db.get(DatasetRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Dataset run not found.")
        db.delete(run)
    return Response(status_code=204)


def _persist_run(config: DatasetConfigIn, summary: dict, stored_rows: list[dict]) -> str | None:
    """Insert a DatasetRun; returns its id, or None if the database is unavailable."""
    from backend.db import DB_AVAILABLE, session_scope

    if not DB_AVAILABLE:
        return None
    from backend.models_db import DatasetRun

    label = (
        f"{summary['business_type']} · {summary['num_users']}u × "
        f"{summary['transactions_per_user']} · seed {summary['random_seed']}"
    )
    try:
        with session_scope() as db:
            run = DatasetRun(
                label=label,
                business_type=summary["business_type"],
                risk_appetite=config.risk_appetite,
                num_users=summary["num_users"],
                transactions_per_user=summary["transactions_per_user"],
                fraud_rate=config.fraud_rate,
                seed=summary["random_seed"],
                total_transactions=summary["total_transactions"],
                fraud_count=summary["fraud_count"],
                fraud_percentage=summary["fraud_percentage"],
                fraud_type_distribution=summary["fraud_type_distribution"],
                stored_count=len(stored_rows),
                transactions=stored_rows,
            )
            db.add(run)
            db.flush()
            return run.id
    except Exception as exc:  # noqa: BLE001 — persistence is best-effort
        logger.warning("Failed to persist dataset run: %s", exc)
        return None
