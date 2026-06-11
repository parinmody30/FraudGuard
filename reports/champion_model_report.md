# FraudLab Champion Model Report

## Champion Model

**XGBoost** (XGBoostClassifier) — selected in Phase 3 (PR-AUC 0.903) and confirmed as the sole production detector in Phase 3.1 (no ensemble).

## Metrics (held-out users)

| metric | value |
|---|---|
| PR-AUC | 0.9028 |
| ROC-AUC | 0.9863 |
| Precision | 0.8304 |
| Recall | 0.8194 |
| F1 | 0.8248 |
| Recall_hard | 0.717 |
| Recall_easy | 0.8696 |
| Recall_medium | 0.8475 |
| Recall@1% / 2% / 5% | 0.1894 / 0.3744 / 0.7974 |

## Decision thresholds

- ALLOW: probability < 0.4
- REVIEW: 0.4 ≤ probability < 0.8
- BLOCK: probability ≥ 0.8

## Confidence bands

- LOW: probability < 0.6
- MEDIUM: 0.6 ≤ probability ≤ 0.85
- HIGH: probability > 0.85

## Dataset

- Simulator config: {'num_users': 400, 'transactions_per_user': 50, 'fraud_rate': 0.06, 'business_type': 'bank', 'risk_appetite': 'medium', 'seed': 42}
- Rows: 21277, users: 400, fraud rate: 0.060018
- Split: {'train_rows': 17050, 'test_rows': 4227, 'test_fraud': 227, 'users_disjoint': True}
- Features after preprocessing: 111

## Artifacts

- Model bundle: `models\champion_model.joblib`
- Metadata: `models\champion_metadata.json`
- Training timestamp: 2026-06-10T11:58:33.861882+00:00

## Recommendation

**Champion model approved for explainability integration.**