# CTGAN Realism Evaluation

- Real (simulator soft) rows: **13315**
- Synthetic (CTGAN) rows: **13315**
- **Verdict: CTGAN NEEDS TUNING**

## Statistical similarity

- Mean continuous KS: **0.119** (lower is better)
- Mean categorical JS divergence: **0.0129**
- Correlation preservation (mean abs diff): **0.1107**
- Mutual-info preservation (mean abs diff): **0.0279**
- Fraud prevalence real/synth: **0.0612 / 0.0873** (abs diff 0.0261)
- Fraud-type distribution JS: **0.0021**

### Per-continuous-feature KS

| feature | KS |
|---|---|
| amount | 0.1162 |
| account_age_days | 0.1052 |
| avg_user_spend | 0.1281 |
| median_user_spend | 0.1632 |
| velocity_1h | 0.0505 |
| velocity_24h | 0.0403 |
| merchant_risk_score | 0.2203 |
| device_risk_score | 0.1282 |

### Per-categorical-feature JS divergence

| feature | JS |
|---|---|
| currency | 0.0 |
| merchant_category | 0.0165 |
| payment_method | 0.0478 |
| channel | 0.0125 |
| device_type | 0.0185 |
| city | 0.0167 |
| state | 0.0235 |
| country | 0.0005 |
| fraud_type | 0.0021 |
| is_foreign | 0.0017 |
| is_fraud | 0.0018 |

## ML utility (TSTR / TRTS)

### logistic

| scenario | precision | recall | f1 | roc_auc | pr_auc |
|---|---|---|---|---|---|
| reference_real_on_real | 0.4327 | 0.8653 | 0.5769 | 0.9621 | 0.7782 |
| TSTR | 0.108 | 0.2279 | 0.1466 | 0.5423 | 0.1047 |
| TRTS | 0.1676 | 0.8908 | 0.2821 | 0.9028 | 0.5572 |

### gradient_boosting

| scenario | precision | recall | f1 | roc_auc | pr_auc |
|---|---|---|---|---|---|
| reference_real_on_real | 0.9362 | 0.7184 | 0.8129 | 0.9882 | 0.9003 |
| TSTR | 0.0971 | 0.1187 | 0.1068 | 0.5029 | 0.0911 |
| TRTS | 0.913 | 0.0258 | 0.0501 | 0.6924 | 0.2089 |

## Recommendation

**CTGAN NEEDS TUNING**

- TSTR ROC-AUC drop vs reference (GBM): 0.4853
- Checks:
  - mean_ks_ok: PASS
  - mean_js_ok: PASS
  - prevalence_ok: PASS
  - fraud_type_ok: PASS
  - tstr_utility_ok: FAIL
- Failing reasons:
  - TSTR ROC-AUC drop 0.4853 > 0.15 (ref 0.9882, tstr 0.5029)