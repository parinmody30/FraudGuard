# FraudLab Phase 3 — Fraud Detection Benchmark

## Datasets

- Config: {'num_users': 400, 'transactions_per_user': 50, 'fraud_rate': 0.06, 'business_type': 'bank', 'risk_appetite': 'medium', 'seed': 42}
- Matched fraud rate: **0.06**
- Simulator: 21277 rows, 400 users, fraud 0.06, test fraud 227, users_disjoint=True
- Hybrid: 21277 rows, 21008 users, fraud 0.06, test fraud 244, users_disjoint=True
- Skipped libraries: ['lightgbm', 'catboost']

## Simulator leaderboard (by PR-AUC)

| rank | model | pr_auc | roc_auc | precision | recall | f1 | recall_easy | recall_medium | recall_hard | recall_at_1pct | recall_at_2pct | recall_at_5pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | xgboost | 0.9028 | 0.9863 | 0.8304 | 0.8194 | 0.8248 | 0.8696 | 0.8475 | 0.717 | 0.1894 | 0.3744 | 0.7974 |
| 2 | random_forest | 0.9002 | 0.9875 | 0.979 | 0.6167 | 0.7568 | 0.8348 | 0.8475 | 0.717 | 0.1894 | 0.37 | 0.8018 |
| 3 | gradient_boosting | 0.8998 | 0.9885 | 0.9379 | 0.7313 | 0.8218 | 0.8435 | 0.8136 | 0.7547 | 0.1894 | 0.3656 | 0.793 |
| 4 | logistic_regression | 0.738 | 0.9642 | 0.3459 | 0.8899 | 0.4982 | 0.6957 | 0.6271 | 0.6981 | 0.1894 | 0.3392 | 0.6476 |
| 5 | lstm | 0.4461 | 0.867 | 0.3032 | 0.5771 | 0.3976 | 0.3652 | 0.4915 | 0.5849 | 0.1454 | 0.2599 | 0.4405 |
| 6 | isolation_forest | 0.2849 | 0.8281 | 0.0 | 0.0 | 0.0 | 0.4696 | 0.3051 | 0.1887 | 0.0881 | 0.1586 | 0.3436 |

## Hybrid leaderboard (by PR-AUC)

| rank | model | pr_auc | roc_auc | precision | recall | f1 | recall_easy | recall_medium | recall_hard | recall_at_1pct | recall_at_2pct | recall_at_5pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lstm | 0.1911 | 0.5641 | 0.0724 | 0.4016 | 0.1227 | 0.0357 | 0.0722 | 0.8 | 0.1189 | 0.127 | 0.1557 |
| 2 | random_forest | 0.1219 | 0.5621 | 0.0 | 0.0 | 0.0 | 0.0268 | 0.0619 | 0.8571 | 0.0738 | 0.1148 | 0.1516 |
| 3 | gradient_boosting | 0.1173 | 0.577 | 0.6429 | 0.0369 | 0.0698 | 0.0179 | 0.0412 | 0.7143 | 0.0574 | 0.0697 | 0.1189 |
| 4 | xgboost | 0.0881 | 0.56 | 0.1185 | 0.0656 | 0.0844 | 0.0268 | 0.0619 | 0.4571 | 0.0287 | 0.0369 | 0.1025 |
| 5 | logistic_regression | 0.0767 | 0.5423 | 0.0709 | 0.4467 | 0.1223 | 0.0536 | 0.1031 | 0.3714 | 0.0246 | 0.0369 | 0.1066 |
| 6 | isolation_forest | 0.0702 | 0.5506 | 0.0 | 0.0 | 0.0 | 0.0536 | 0.0619 | 0.1143 | 0.0205 | 0.0246 | 0.0615 |

## Hybrid - Simulator improvement

| model | sim_pr_auc | hyb_pr_auc | delta_pr_auc | sim_recall_hard | hyb_recall_hard | delta_recall_hard |
|---|---|---|---|---|---|---|
| logistic_regression | 0.738 | 0.0767 | -0.6613 | 0.6981 | 0.3714 | -0.3267 |
| random_forest | 0.9002 | 0.1219 | -0.7783 | 0.717 | 0.8571 | 0.1401 |
| gradient_boosting | 0.8998 | 0.1173 | -0.7825 | 0.7547 | 0.7143 | -0.0404 |
| xgboost | 0.9028 | 0.0881 | -0.8147 | 0.717 | 0.4571 | -0.2599 |
| isolation_forest | 0.2849 | 0.0702 | -0.2147 | 0.1887 | 0.1143 | -0.0744 |
| lstm | 0.4461 | 0.1911 | -0.255 | 0.5849 | 0.8 | 0.2151 |

## Sequential (LSTM) findings — recall_hard

- **simulator**: LSTM recall_hard=0.5849 vs best row-level (xgboost) recall_hard=0.717 → Δ=-0.1321 (LSTM helps: False)
- **hybrid**: LSTM recall_hard=0.8 vs best row-level (random_forest) recall_hard=0.8571 → Δ=-0.0571 (LSTM helps: False)

## Decision

- PR-AUC deltas: {'n': 6, 'improved': 0, 'degraded': 6, 'neutral': 0, 'median_delta': -0.7198}
- recall_hard deltas: {'n': 6, 'improved': 2, 'degraded': 4, 'neutral': 0, 'median_delta': -0.0574}

**RECOMMENDATION (C): ABANDON CTGAN FOR MODEL TRAINING; USE SIMULATOR ONLY**