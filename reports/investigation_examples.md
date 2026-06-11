# FraudLab Investigation Agent - Examples (Phase 6)

Generated in **Groq (llama-3.3-70b-versatile)** mode. The agent is grounded only in champion XGBoost outputs, decision thresholds, confidence bands, and SHAP contributors; recommendation and next steps come from the decision policy.

## Example: BLOCK fraud

Actual label: FRAUD - account_takeover

Explanation input (champion + SHAP):
```json
{
  "transaction_id": "T0000020057",
  "fraud_probability": 1.0,
  "decision": "BLOCK",
  "confidence": "HIGH",
  "contributors": [
    {
      "feature": "Device Risk",
      "impact": 6.8696,
      "direction": "increase"
    },
    {
      "feature": "Time of Day",
      "impact": 2.998,
      "direction": "increase"
    },
    {
      "feature": "Transaction State",
      "impact": 2.3321,
      "direction": "increase"
    },
    {
      "feature": "Merchant Category",
      "impact": 1.4144,
      "direction": "increase"
    },
    {
      "feature": "Velocity Spike (1 Hour)",
      "impact": 1.1554,
      "direction": "increase"
    },
    {
      "feature": "Transaction City",
      "impact": -0.2231,
      "direction": "decrease"
    },
    {
      "feature": "Account Age",
      "impact": -0.1447,
      "direction": "decrease"
    },
    {
      "feature": "Transaction Amount",
      "impact": -0.1319,
      "direction": "decrease"
    }
  ]
}
```

Investigation narrative:
- **Summary:** The transaction was blocked due to a high fraud probability, driven by multiple increasing risk factors. The decision was made with high confidence.
- **Signals detected:**
  - High Device Risk
  - Unusual Time of Day
  - Suspicious Transaction State
  - High-Risk Merchant Category
  - Recent Velocity Spike
- **Recommendation:** Prevent execution and initiate fraud protocols.
- **Confidence explanation:** The high confidence level is supported by the strong impact of multiple increasing risk factors, including Device Risk and Time of Day, which contributed significantly to the high fraud probability.
- **Next steps:**
  - Suspend transaction processing and initiate account review.
  - Initiate fraud protocols and notify the customer.

Frontend response:
```json
{
  "transaction_id": "T0000020057",
  "summary": "The transaction was blocked due to a high fraud probability, driven by multiple increasing risk factors. The decision was made with high confidence.",
  "signals_detected": [
    "High Device Risk",
    "Unusual Time of Day",
    "Suspicious Transaction State",
    "High-Risk Merchant Category",
    "Recent Velocity Spike"
  ],
  "recommendation": "Prevent execution and initiate fraud protocols.",
  "confidence_explanation": "The high confidence level is supported by the strong impact of multiple increasing risk factors, including Device Risk and Time of Day, which contributed significantly to the high fraud probability.",
  "next_steps": [
    "Suspend transaction processing and initiate account review.",
    "Initiate fraud protocols and notify the customer."
  ]
}
```

## Example: REVIEW fraud

Actual label: FRAUD - geo_anomaly

Explanation input (champion + SHAP):
```json
{
  "transaction_id": "T0000020290",
  "fraud_probability": 0.764428,
  "decision": "REVIEW",
  "confidence": "MEDIUM",
  "contributors": [
    {
      "feature": "Transaction State",
      "impact": 3.9867,
      "direction": "increase"
    },
    {
      "feature": "Account Age",
      "impact": -2.1998,
      "direction": "decrease"
    },
    {
      "feature": "Velocity (24 Hour)",
      "impact": 2.0016,
      "direction": "increase"
    },
    {
      "feature": "Device Risk",
      "impact": -1.5504,
      "direction": "decrease"
    },
    {
      "feature": "Median Spend",
      "impact": -0.4616,
      "direction": "decrease"
    },
    {
      "feature": "Merchant Risk",
      "impact": 0.422,
      "direction": "increase"
    },
    {
      "feature": "Country",
      "impact": 0.3079,
      "direction": "increase"
    },
    {
      "feature": "Average Spend",
      "impact": 0.2543,
      "direction": "increase"
    }
  ]
}
```

Investigation narrative:
- **Summary:** The transaction has been flagged for review due to a medium confidence level in its potential fraudulence, with a probability of 0.764428. The decision to review is based on various contributing factors.
- **Signals detected:**
  - High Transaction State
  - Low Account Age
  - High Velocity (24 Hour)
  - Low Device Risk
  - Low Median Spend
  - High Merchant Risk
  - Country
  - High Average Spend
- **Recommendation:** Escalate for analyst verification.
- **Confidence explanation:** The medium confidence level in the review decision is supported by the fraud probability of 0.764428 and the presence of strong contributing factors, such as Transaction State and Velocity (24 Hour), which increase the likelihood of fraud.
- **Next steps:**
  - Verify transaction legitimacy with the customer.
  - Confirm the device and location with the account holder.

Frontend response:
```json
{
  "transaction_id": "T0000020290",
  "summary": "The transaction has been flagged for review due to a medium confidence level in its potential fraudulence, with a probability of 0.764428. The decision to review is based on various contributing factors.",
  "signals_detected": [
    "High Transaction State",
    "Low Account Age",
    "High Velocity (24 Hour)",
    "Low Device Risk",
    "Low Median Spend",
    "High Merchant Risk",
    "Country",
    "High Average Spend"
  ],
  "recommendation": "Escalate for analyst verification.",
  "confidence_explanation": "The medium confidence level in the review decision is supported by the fraud probability of 0.764428 and the presence of strong contributing factors, such as Transaction State and Velocity (24 Hour), which increase the likelihood of fraud.",
  "next_steps": [
    "Verify transaction legitimacy with the customer.",
    "Confirm the device and location with the account holder."
  ]
}
```

## Example: ALLOW legitimate

Actual label: legitimate

Explanation input (champion + SHAP):
```json
{
  "transaction_id": "T0000018450",
  "fraud_probability": 1e-06,
  "decision": "ALLOW",
  "confidence": "LOW",
  "contributors": [
    {
      "feature": "Transaction Amount",
      "impact": -3.1748,
      "direction": "decrease"
    },
    {
      "feature": "Velocity (24 Hour)",
      "impact": -2.8098,
      "direction": "decrease"
    },
    {
      "feature": "Merchant Category",
      "impact": -1.8039,
      "direction": "decrease"
    },
    {
      "feature": "Device Type",
      "impact": 0.0853,
      "direction": "increase"
    }
  ]
}
```

Investigation narrative:
- **Summary:** The transaction was allowed due to a low fraud probability, despite the model having low confidence in this decision. The low confidence suggests that the signals detected were not strongly indicative of a legitimate transaction.
- **Signals detected:**
  - Low Transaction Amount
  - Low Velocity (24 Hour)
  - Unusual Merchant Category
  - Common Device Type
- **Recommendation:** Proceed without intervention.
- **Confidence explanation:** The confidence is low because, although the fraud probability is very low, the SHAP contributors, while mostly decreasing the fraud probability, do not strongly support the allow decision.
- **Next steps:**
  - Monitor for recurrence.

Frontend response:
```json
{
  "transaction_id": "T0000018450",
  "summary": "The transaction was allowed due to a low fraud probability, despite the model having low confidence in this decision. The low confidence suggests that the signals detected were not strongly indicative of a legitimate transaction.",
  "signals_detected": [
    "Low Transaction Amount",
    "Low Velocity (24 Hour)",
    "Unusual Merchant Category",
    "Common Device Type"
  ],
  "recommendation": "Proceed without intervention.",
  "confidence_explanation": "The confidence is low because, although the fraud probability is very low, the SHAP contributors, while mostly decreasing the fraud probability, do not strongly support the allow decision.",
  "next_steps": [
    "Monitor for recurrence."
  ]
}
```

## Approval

Investigation Agent approved for analyst workflows.
