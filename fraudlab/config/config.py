"""FraudLabConfig and supporting enums.

The config is the single source of truth for a generation run. Combined with the
``random_seed`` it makes generation fully deterministic and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any


class BusinessType(str, Enum):
    BANK = "bank"
    CREDIT_CARD = "credit_card"
    UPI = "upi"
    ECOMMERCE = "ecommerce"


class RiskAppetite(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class OutputFormat(str, Enum):
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"


# Allowed payment methods and channels per business type. The first payment method is
# treated as the user's default rail for that business.
BUSINESS_PAYMENT_METHODS: dict[BusinessType, tuple[str, ...]] = {
    BusinessType.BANK: ("upi", "imps", "neft", "debit_card"),
    BusinessType.CREDIT_CARD: ("credit_card",),
    BusinessType.UPI: ("upi",),
    BusinessType.ECOMMERCE: ("upi", "credit_card", "debit_card", "wallet", "netbanking", "cod"),
}

BUSINESS_CHANNELS: dict[BusinessType, tuple[str, ...]] = {
    BusinessType.BANK: ("mobile", "netbanking", "branch", "atm"),
    BusinessType.CREDIT_CARD: ("pos", "online", "contactless"),
    BusinessType.UPI: ("mobile", "qr"),
    BusinessType.ECOMMERCE: ("web", "app"),
}

# All fraud scenarios grouped by difficulty tier. Used to build/validate fraud_mix.
FRAUD_TIERS: dict[str, tuple[str, ...]] = {
    "easy": (
        "account_takeover",
        "large_spend",
        "geo_anomaly",
        "night_time",
        "new_device",
    ),
    "medium": (
        "card_testing",
        "velocity",
        "merchant",
        "beneficiary",
    ),
    "hard": (
        "synthetic_identity",
        "money_mule",
        "gradual_escalation",
        "cross_channel",
        "multi_stage",
    ),
}

ALL_FRAUD_TYPES: tuple[str, ...] = tuple(
    name for tier in FRAUD_TIERS.values() for name in tier
)

# Default weighting across difficulty tiers per risk appetite. A higher risk appetite
# shifts the mix toward subtle (hard) fraud that mimics legitimate behavior.
RISK_APPETITE_TIER_WEIGHTS: dict[RiskAppetite, dict[str, float]] = {
    RiskAppetite.LOW: {"easy": 0.65, "medium": 0.25, "hard": 0.10},
    RiskAppetite.MEDIUM: {"easy": 0.40, "medium": 0.35, "hard": 0.25},
    RiskAppetite.HIGH: {"easy": 0.20, "medium": 0.35, "hard": 0.45},
}


def _coerce_enum(value: Any, enum_cls: type[Enum]) -> Any:
    """Accept either an enum member or its string value."""
    if isinstance(value, enum_cls):
        return value
    return enum_cls(value)


def default_fraud_mix(risk_appetite: RiskAppetite) -> dict[str, float]:
    """Build a per-scenario fraud mix from the tier weights for a risk appetite.

    Each tier's weight is spread evenly across its scenarios, then the whole mapping is
    normalized to sum to 1.0.
    """
    tier_weights = RISK_APPETITE_TIER_WEIGHTS[risk_appetite]
    mix: dict[str, float] = {}
    for tier, scenarios in FRAUD_TIERS.items():
        share = tier_weights[tier] / len(scenarios)
        for name in scenarios:
            mix[name] = share
    return _normalize_mix(mix)


def _normalize_mix(mix: dict[str, float]) -> dict[str, float]:
    total = sum(mix.values())
    if total <= 0:
        raise ValueError("fraud_mix weights must sum to a positive value")
    return {k: v / total for k, v in mix.items()}


@dataclass(slots=True)
class FraudLabConfig:
    """Top-level configuration for a dataset generation run."""

    num_users: int = 500
    transactions_per_user: int = 100
    fraud_rate: float = 0.02
    fraud_mix: dict[str, float] | None = None
    business_type: BusinessType = BusinessType.BANK
    noise_level: float = 0.3
    risk_appetite: RiskAppetite = RiskAppetite.MEDIUM
    random_seed: int = 42
    output_format: OutputFormat = OutputFormat.CSV
    start_date: date = field(default_factory=lambda: date(2024, 1, 1))
    currency: str = "INR"

    def __post_init__(self) -> None:
        # Coerce string inputs (e.g. from YAML) into enums.
        self.business_type = _coerce_enum(self.business_type, BusinessType)
        self.risk_appetite = _coerce_enum(self.risk_appetite, RiskAppetite)
        self.output_format = _coerce_enum(self.output_format, OutputFormat)
        if isinstance(self.start_date, str):
            self.start_date = date.fromisoformat(self.start_date)

        # Validate numeric ranges.
        if self.num_users < 1:
            raise ValueError("num_users must be >= 1")
        if self.transactions_per_user < 1:
            raise ValueError("transactions_per_user must be >= 1")
        if not 0.0 <= self.fraud_rate <= 1.0:
            raise ValueError("fraud_rate must be in [0, 1]")
        if not 0.0 <= self.noise_level <= 1.0:
            raise ValueError("noise_level must be in [0, 1]")

        # Build or validate the fraud mix.
        if self.fraud_mix is None:
            self.fraud_mix = default_fraud_mix(self.risk_appetite)
        else:
            unknown = set(self.fraud_mix) - set(ALL_FRAUD_TYPES)
            if unknown:
                raise ValueError(f"Unknown fraud types in fraud_mix: {sorted(unknown)}")
            if any(w < 0 for w in self.fraud_mix.values()):
                raise ValueError("fraud_mix weights must be non-negative")
            self.fraud_mix = _normalize_mix(self.fraud_mix)

    # --- convenience accessors -------------------------------------------------

    @property
    def total_transactions(self) -> int:
        return self.num_users * self.transactions_per_user

    @property
    def payment_methods(self) -> tuple[str, ...]:
        return BUSINESS_PAYMENT_METHODS[self.business_type]

    @property
    def channels(self) -> tuple[str, ...]:
        return BUSINESS_CHANNELS[self.business_type]

    # --- loaders ---------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FraudLabConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str) -> "FraudLabConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)
