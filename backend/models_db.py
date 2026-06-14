"""SQLAlchemy ORM models for persisted dataset runs.

A ``DatasetRun`` captures one dataset generation: its config, summary statistics, and a capped
sample of transaction rows (stored as JSONB) so the Dashboard and Transactions table can be
driven from history without regenerating. ``fraud_type_distribution`` and ``transactions`` use
Postgres JSONB columns.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.db import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class DatasetRun(Base):
    __tablename__ = "dataset_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    label: Mapped[str] = mapped_column(String(120))

    # config
    business_type: Mapped[str] = mapped_column(String(32))
    risk_appetite: Mapped[str] = mapped_column(String(16))
    num_users: Mapped[int] = mapped_column(Integer)
    transactions_per_user: Mapped[int] = mapped_column(Integer)
    fraud_rate: Mapped[float] = mapped_column(Float)
    seed: Mapped[int] = mapped_column(Integer)

    # summary
    total_transactions: Mapped[int] = mapped_column(Integer)
    fraud_count: Mapped[int] = mapped_column(Integer)
    fraud_percentage: Mapped[float] = mapped_column(Float)
    fraud_type_distribution: Mapped[dict[str, int]] = mapped_column(JSONB, default=dict)

    # stored rows (capped sample)
    stored_count: Mapped[int] = mapped_column(Integer, default=0)
    transactions: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)

    def as_list_item(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "label": self.label,
            "business_type": self.business_type,
            "risk_appetite": self.risk_appetite,
            "num_users": self.num_users,
            "transactions_per_user": self.transactions_per_user,
            "fraud_rate": self.fraud_rate,
            "seed": self.seed,
            "total_transactions": self.total_transactions,
            "fraud_count": self.fraud_count,
            "fraud_percentage": self.fraud_percentage,
            "stored_count": self.stored_count,
        }

    def as_detail(self) -> dict[str, Any]:
        return {
            **self.as_list_item(),
            "fraud_type_distribution": self.fraud_type_distribution or {},
            "transactions": self.transactions or [],
        }
