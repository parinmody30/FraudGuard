# CTGAN Realism Evaluation

- Real (simulator soft) rows: **13315**
- Synthetic (CTGAN) rows: **13315**
- **Verdict: CTGAN NEEDS TUNING**

## Statistical similarity

- Mean continuous KS: **0.1057** (lower is better)
- Mean categorical JS divergence: **0.0367**
- Correlation preservation (mean abs diff): **0.0902**
- Mutual-info preservation (mean abs diff): **0.0624**
- Fraud prevalence real/synth: **0.0612 / 0.246** (abs diff 0.1848)
- Fraud-type distribution JS: **0.059**

### Per-continuous-feature KS

| feature | KS |
|---|---|
| amount | 0.127 |
| account_age_days | 0.0391 |
| avg_user_spend | 0.0809 |
| median_user_spend | 0.0935 |
| velocity_1h | 0.0975 |
| velocity_24h | 0.2211 |
| merchant_risk_score | 0.1103 |
| device_risk_score | 0.0759 |

### Per-categorical-feature JS divergence

| feature | JS |
|---|---|
| currency | 0.0 |
| merchant_category | 0.0209 |
| payment_method | 0.0199 |
| channel | 0.0181 |
| device_type | 0.0119 |
| city | 0.0618 |
| state | 0.0623 |
| country | 0.0501 |
| fraud_type | 0.059 |
| is_foreign | 0.0492 |
| is_fraud | 0.0502 |

## ML utility (TSTR / TRTS)

### logistic

| scenario | precision | recall | f1 | roc_auc | pr_auc |
|---|---|---|---|---|---|
| reference_real_on_real | 0.4327 | 0.8653 | 0.5769 | 0.9621 | 0.7782 |
| TSTR | 0.6496 | 0.8299 | 0.7288 | 0.9114 | 0.7846 |
| TRTS | 0.5924 | 0.7276 | 0.6531 | 0.954 | 0.7389 |

### gradient_boosting

| scenario | precision | recall | f1 | roc_auc | pr_auc |
|---|---|---|---|---|---|
| reference_real_on_real | 0.9362 | 0.7184 | 0.8129 | 0.9882 | 0.9003 |
| TSTR | 0.7236 | 0.6843 | 0.7034 | 0.8868 | 0.761 |
| TRTS | 0.5941 | 0.5926 | 0.5934 | 0.9483 | 0.6414 |

## Recommendation

**CTGAN NEEDS TUNING**

- TSTR ROC-AUC drop vs reference (GBM): 0.1014
- Checks:
  - mean_ks_ok: PASS
  - mean_js_ok: PASS
  - prevalence_ok: FAIL
  - fraud_type_ok: PASS
  - tstr_utility_ok: PASS
- Failing reasons:
  - fraud-prevalence abs diff 0.1848 > 0.03