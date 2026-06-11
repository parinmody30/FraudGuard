# FraudLab Phase 3.1 - Ensemble Feasibility Study

- Test rows: 4227, test fraud: 227, users_disjoint: True
- Voting weights (PR-AUC normalized): {'logistic_regression': 0.738, 'random_forest': 0.9002, 'gradient_boosting': 0.8998, 'xgboost': 0.9028}

## 1. Confusion matrices (Phase 3 operating threshold)

| model | TP | FP | TN | FN |
|---|---|---|---|---|
| logistic_regression | 202 | 382 | 3618 | 25 |
| random_forest | 140 | 3 | 3997 | 87 |
| gradient_boosting | 166 | 11 | 3989 | 61 |
| xgboost | 186 | 38 | 3962 | 41 |

## 2. TP overlap matrix (Jaccard)

| | logistic_regression | random_forest | gradient_boosting | xgboost |
|---|---|---|---|---|
| logistic_regression | 1.0 | 0.6683 | 0.7951 | 0.8216 |
| random_forest | 0.6683 | 1.0 | 0.8323 | 0.7249 |
| gradient_boosting | 0.7951 | 0.8323 | 1.0 | 0.8624 |
| xgboost | 0.8216 | 0.7249 | 0.8624 | 1.0 |

## 3. FN recovery (% of a model's missed frauds recovered by another)

| | logistic_regression | random_forest | gradient_boosting | xgboost |
|---|---|---|---|---|
| logistic_regression | 0.0 | 12.0 | 12.0 | 44.0 |
| random_forest | 74.71 | 0.0 | 31.03 | 56.32 |
| gradient_boosting | 63.93 | 1.64 | 0.0 | 37.7 |
| xgboost | 65.85 | 7.32 | 7.32 | 0.0 |

Recovered by ANY other model:
- logistic_regression: missed 25, recovered by any = 48.0%
- random_forest: missed 87, recovered by any = 85.06%
- gradient_boosting: missed 61, recovered by any = 78.69%
- xgboost: missed 41, recovered by any = 68.29%

## 4. Fraud-tier residuals (TP / FN per tier)

| model | easy TP | easy FN | medium TP | medium FN | hard TP | hard FN |
|---|---|---|---|---|---|---|
| logistic_regression | 103 | 12 | 53 | 6 | 46 | 7 |
| random_forest | 75 | 40 | 32 | 27 | 33 | 20 |
| gradient_boosting | 88 | 27 | 43 | 16 | 35 | 18 |
| xgboost | 99 | 16 | 49 | 10 | 38 | 15 |

## 5. Predicted-probability correlation (Pearson)

| | logistic_regression | random_forest | gradient_boosting | xgboost |
|---|---|---|---|---|
| logistic_regression | 1.0 | 0.73 | 0.6537 | 0.6555 |
| random_forest | 0.73 | 1.0 | 0.9469 | 0.9182 |
| gradient_boosting | 0.6537 | 0.9469 | 1.0 | 0.9273 |
| xgboost | 0.6555 | 0.9182 | 0.9273 | 1.0 |

## 6. Ensemble benchmark

| model | roc_auc | pr_auc | precision | recall | f1 | recall_easy | recall_medium | recall_hard | recall_at_1pct | recall_at_2pct | recall_at_5pct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| xgboost | 0.9863 | 0.9028 | 0.8304 | 0.8194 | 0.8248 | 0.8696 | 0.8475 | 0.717 | 0.1894 | 0.3744 | 0.7974 |
| soft_voting | 0.986 | 0.9068 | 0.8282 | 0.8282 | 0.8282 | 0.8609 | 0.8475 | 0.7358 | 0.1894 | 0.3744 | 0.8194 |
| weighted_voting | 0.9866 | 0.9094 | 0.8238 | 0.8238 | 0.8238 | 0.8522 | 0.8475 | 0.7358 | 0.1894 | 0.3744 | 0.8238 |
| stacking | 0.9874 | 0.913 | 0.8194 | 0.8194 | 0.8194 | 0.8696 | 0.8305 | 0.6981 | 0.1894 | 0.3744 | 0.8062 |

## 7. Decision

- Reference XGBoost: PR-AUC=0.9028, recall_hard=0.717
- soft_voting: PR-AUC=0.9068 (Δ 0.004), recall_hard=0.7358 (Δ 0.0188)
- weighted_voting: PR-AUC=0.9094 (Δ 0.0066), recall_hard=0.7358 (Δ 0.0188)
- stacking: PR-AUC=0.913 (Δ 0.0102), recall_hard=0.6981 (Δ -0.0189)

**RECOMMENDATION (B): USE XGBOOST ONLY; AVOID ENSEMBLE COMPLEXITY**