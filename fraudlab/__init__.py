"""FraudLab Generator — synthetic financial-transaction dataset generator.

Phase 1: population simulation, behavioral baseline transactions, a multi-difficulty
fraud-injection engine, deterministic configuration, and a multi-format export pipeline.

This package generates datasets only. It is intentionally NOT a detector, API, dashboard,
or evaluation framework.
"""

from fraudlab.config.config import (
    BusinessType,
    FraudLabConfig,
    OutputFormat,
    RiskAppetite,
)
from fraudlab.pipeline import build_dataset

__all__ = [
    "BusinessType",
    "FraudLabConfig",
    "OutputFormat",
    "RiskAppetite",
    "build_dataset",
]

__version__ = "0.1.0"
