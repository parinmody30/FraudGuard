"""Human-friendly labels and deterministic analyst summaries for SHAP explanations.

Raw feature-engineering names (e.g. ``velocity_1h``, ``categorical__city_Mumbai``) are never
shown to analysts. This module maps each *base* model feature to an analyst-friendly label
and renders the positive/negative contributor summary. Fully deterministic - no LLMs.
"""

from __future__ import annotations

from typing import Any

# Base model feature -> analyst-friendly label. Keys are the original schema-derived
# feature names produced by the Phase 3 preprocessing (before one-hot expansion).
FEATURE_LABELS: dict[str, str] = {
    "amount": "Transaction Amount",
    "account_age_days": "Account Age",
    "avg_user_spend": "Average Spend",
    "median_user_spend": "Median Spend",
    "velocity_1h": "Velocity Spike (1 Hour)",
    "velocity_24h": "Velocity (24 Hour)",
    "merchant_risk_score": "Merchant Risk",
    "device_risk_score": "Device Risk",
    "hour": "Time of Day",
    "currency": "Currency",
    "merchant_category": "Merchant Category",
    "payment_method": "Payment Method",
    "channel": "Channel",
    "device_type": "Device Type",
    "city": "Transaction City",
    "state": "Transaction State",
    "country": "Country",
    "day_of_week": "Day of Week",
    "is_foreign": "Foreign Transaction",
    "is_weekend": "Weekend Transaction",
}


def friendly_label(base_feature: str) -> str:
    """Return the analyst-friendly label for a base feature (title-cased fallback)."""
    if base_feature in FEATURE_LABELS:
        return FEATURE_LABELS[base_feature]
    return base_feature.replace("_", " ").title()


def format_analyst_summary(explanation: dict[str, Any]) -> str:
    """Render a deterministic positive/negative contributor summary for one explanation."""
    contributors = explanation.get("contributors", [])
    positives = [c for c in contributors if c["direction"] == "increase"]
    negatives = [c for c in contributors if c["direction"] == "decrease"]

    lines: list[str] = []
    decision = explanation.get("decision", "?")
    prob = explanation.get("fraud_probability", 0.0)
    conf = explanation.get("confidence", "?")
    lines.append(
        f"Transaction {explanation.get('transaction_id', '?')}: {decision} "
        f"(fraud probability {prob}, {conf} confidence)."
    )
    lines.append("")
    lines.append("Positive Contributors:")
    if positives:
        for c in positives:
            lines.append(f"* {c['feature']} (+{abs(c['impact'])})")
    else:
        lines.append("* (none)")
    lines.append("")
    lines.append("Negative Contributors:")
    if negatives:
        for c in negatives:
            lines.append(f"* {c['feature']} (-{abs(c['impact'])})")
    else:
        lines.append("* (none)")
    return "\n".join(lines)
