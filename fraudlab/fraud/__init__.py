"""Fraud scenario engine.

Importing this package registers all built-in scenarios (easy/medium/hard) into the
shared REGISTRY via their ``@register`` decorators.
"""

from fraudlab.fraud import easy, hard, medium  # noqa: F401  (import for registration)
from fraudlab.fraud.base import REGISTRY, FraudContext, FraudScenario
from fraudlab.fraud.engine import FraudEngine

__all__ = [
    "REGISTRY",
    "FraudContext",
    "FraudEngine",
    "FraudScenario",
]
