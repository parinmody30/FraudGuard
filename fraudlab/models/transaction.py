"""Transaction model and the canonical schema field order."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

# Canonical column order for the exported dataset. Exporters reindex against this list
# to guarantee schema integrity across CSV / JSON / Parquet.
TRANSACTION_FIELDS: tuple[str, ...] = (
    "transaction_id",
    "user_id",
    "timestamp",
    "amount",
    "currency",
    "merchant_id",
    "merchant_name",
    "merchant_category",
    "payment_method",
    "channel",
    "device_id",
    "device_type",
    "city",
    "state",
    "country",
    "is_foreign",
    "account_age_days",
    "avg_user_spend",
    "median_user_spend",
    "velocity_1h",
    "velocity_24h",
    "merchant_risk_score",
    "device_risk_score",
    "is_fraud",
    "fraud_type",
)


@dataclass(slots=True)
class Transaction:
    """A single financial transaction.

    The rolling features (``avg_user_spend``, ``median_user_spend``, ``velocity_1h``,
    ``velocity_24h``) are computed causally — as-of the transaction timestamp using only
    prior transactions — so they are leakage-free features a detector could legitimately
    use at scoring time.
    """

    transaction_id: str
    user_id: str
    timestamp: datetime

    amount: float
    currency: str

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

    is_fraud: bool = False
    fraud_type: str | None = None

    def to_row(self) -> dict[str, Any]:
        """Return an ordered dict matching ``TRANSACTION_FIELDS``."""
        data = asdict(self)
        return {field: data[field] for field in TRANSACTION_FIELDS}
