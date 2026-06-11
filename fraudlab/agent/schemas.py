"""Input/output contracts and policy text for the Investigation Agent (Phase 6).

The agent consumes a Phase 5 explanation object and returns a five-section investigation
report. ``recommendation`` and ``next_steps`` are derived *only* from the decision policy;
``confidence_explanation`` only from the confidence band; ``signals_detected`` only from the
SHAP contributors. Validation here guarantees the agent never operates on malformed or
missing evidence.
"""

from __future__ import annotations

from typing import Any

VALID_DECISIONS = ("ALLOW", "REVIEW", "BLOCK")
VALID_CONFIDENCE = ("LOW", "MEDIUM", "HIGH")

# Mandatory keys on the explanation input and the investigation output.
REQUIRED_INPUT_KEYS = ("fraud_probability", "decision", "confidence", "contributors")
REPORT_FIELDS = (
    "summary",
    "signals_detected",
    "recommendation",
    "confidence_explanation",
    "next_steps",
)
FRONTEND_FIELDS = ("transaction_id", *REPORT_FIELDS)

# Recommendation text is a pure function of the decision (decision policy only).
RECOMMENDATION_BY_DECISION: dict[str, str] = {
    "ALLOW": "Proceed without intervention.",
    "REVIEW": "Escalate for analyst verification.",
    "BLOCK": "Prevent execution and initiate fraud protocols.",
}

# Next steps are a pure function of the decision.
NEXT_STEPS_BY_DECISION: dict[str, list[str]] = {
    "ALLOW": ["Monitor for recurrence."],
    "REVIEW": [
        "Verify transaction legitimacy with the customer.",
        "Confirm the device and location with the account holder.",
    ],
    "BLOCK": [
        "Suspend transaction processing and initiate account review.",
        "Initiate fraud protocols and notify the customer.",
    ],
}


class ExplanationValidationError(ValueError):
    """Raised when an explanation input violates the mandatory contract."""


def validate_explanation(explanation: dict[str, Any]) -> dict[str, Any]:
    """Validate + normalize a Phase 5 explanation into the agent's grounded evidence.

    Returns a dict with: ``transaction_id``, ``fraud_probability``, ``decision``,
    ``confidence``, ``contributors``, ``transaction`` (safe context subset).
    """
    if not isinstance(explanation, dict):
        raise ExplanationValidationError("explanation must be a dict")

    missing = [k for k in REQUIRED_INPUT_KEYS if k not in explanation]
    if missing:
        raise ExplanationValidationError(f"missing required keys: {missing}")

    decision = str(explanation["decision"]).upper()
    confidence = str(explanation["confidence"]).upper()
    if decision not in VALID_DECISIONS:
        raise ExplanationValidationError(f"invalid decision: {decision}")
    if confidence not in VALID_CONFIDENCE:
        raise ExplanationValidationError(f"invalid confidence: {confidence}")

    try:
        prob = float(explanation["fraud_probability"])
    except (TypeError, ValueError) as exc:
        raise ExplanationValidationError("fraud_probability must be numeric") from exc

    contributors = explanation.get("contributors") or []
    if not isinstance(contributors, list):
        raise ExplanationValidationError("contributors must be a list")
    norm_contribs: list[dict[str, Any]] = []
    for c in contributors:
        if not isinstance(c, dict) or "feature" not in c or "direction" not in c:
            raise ExplanationValidationError("each contributor needs feature + direction")
        norm_contribs.append(
            {
                "feature": str(c["feature"]),
                "impact": float(c.get("impact", 0.0)),
                "direction": str(c["direction"]).lower(),
            }
        )

    transaction = explanation.get("transaction") or {}
    txn_id = (
        explanation.get("transaction_id")
        or (transaction.get("transaction_id") if isinstance(transaction, dict) else None)
        or "unknown"
    )

    return {
        "transaction_id": str(txn_id),
        "fraud_probability": prob,
        "decision": decision,
        "confidence": confidence,
        "contributors": norm_contribs,
        "transaction": _safe_transaction(transaction),
    }


# Transaction fields that are safe, non-identifying context to forward to the LLM.
_SAFE_TXN_FIELDS = (
    "amount", "currency", "merchant_category", "payment_method", "channel",
    "device_type", "city", "state", "country", "is_foreign", "timestamp",
)


def _safe_transaction(transaction: Any) -> dict[str, Any]:
    """Return only safe descriptive context (never ids/artifacts/raw feature names)."""
    if not isinstance(transaction, dict):
        return {}
    return {k: transaction[k] for k in _SAFE_TXN_FIELDS if k in transaction}


def is_valid_report(report: dict[str, Any]) -> bool:
    """True if ``report`` satisfies the five-section output contract."""
    if not isinstance(report, dict):
        return False
    if any(k not in report for k in REPORT_FIELDS):
        return False
    return (
        isinstance(report["summary"], str)
        and isinstance(report["signals_detected"], list)
        and isinstance(report["recommendation"], str)
        and isinstance(report["confidence_explanation"], str)
        and isinstance(report["next_steps"], list)
    )
