# FraudLab SHAP Explanation Examples (Phase 5)

Champion model: **XGBoost** (`models/champion_model.joblib`). Explainer: `shap.TreeExplainer` (no retraining). One-hot columns are aggregated to base features; raw feature-engineering names are never shown.

## Example 1 - High-risk fraud (BLOCK)

- transaction_id: `T0000020057`  (actual: FRAUD - account_takeover)
- fraud_probability: **1.0**
- decision: **BLOCK**
- confidence: **HIGH**

Contributors (SHAP, log-odds):

| feature | impact | direction |
|---|---|---|
| Device Risk | +6.8696 | increase |
| Time of Day | +2.9980 | increase |
| Transaction State | +2.3321 | increase |
| Merchant Category | +1.4144 | increase |
| Velocity Spike (1 Hour) | +1.1554 | increase |
| Transaction City | -0.2231 | decrease |
| Account Age | -0.1447 | decrease |
| Transaction Amount | -0.1319 | decrease |

Frontend response:
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

Analyst summary:
```
Transaction T0000020057: BLOCK (fraud probability 1.0, HIGH confidence).

Positive Contributors:
* Device Risk (+6.8696)
* Time of Day (+2.998)
* Transaction State (+2.3321)
* Merchant Category (+1.4144)
* Velocity Spike (1 Hour) (+1.1554)

Negative Contributors:
* Transaction City (-0.2231)
* Account Age (-0.1447)
* Transaction Amount (-0.1319)
```

## Example 2 - Legitimate transaction (ALLOW)

- transaction_id: `T0000018450`  (actual: legitimate)
- fraud_probability: **1e-06**
- decision: **ALLOW**
- confidence: **LOW**

Contributors (SHAP, log-odds):

| feature | impact | direction |
|---|---|---|
| Transaction Amount | -3.1748 | decrease |
| Velocity (24 Hour) | -2.8098 | decrease |
| Merchant Category | -1.8039 | decrease |
| Device Type | +0.0853 | increase |

Frontend response:
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

Analyst summary:
```
Transaction T0000018450: ALLOW (fraud probability 1e-06, LOW confidence).

Positive Contributors:
* Device Type (+0.0853)

Negative Contributors:
* Transaction Amount (-3.1748)
* Velocity (24 Hour) (-2.8098)
* Merchant Category (-1.8039)
```

## Approval

"Champion model explanations approved for investigation workflows."
