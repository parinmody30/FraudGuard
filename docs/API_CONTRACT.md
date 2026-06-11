# FraudLab API Contract (Phase 7.0)

Single source of truth for the backend ↔ frontend integration. Every endpoint below is a
thin wrapper over an already-implemented FraudLab module — **no new business logic**.

- Base URL: `http://localhost:8000` (configurable via `VITE_API_BASE_URL`)
- Content type: `application/json`
- Backend app: `backend/app.py` (`uvicorn backend.app:app --reload`)
- Pydantic schemas: `backend/api/schemas.py`
- Frontend services: `frontend/src/services/`

## Backed-by map

| Capability | Backend module reused | Phase |
|---|---|---|
| Scoring | `fraudlab.models.predictor.ChampionPredictor` | 4 |
| Explainability | `fraudlab.explainability.shap_explainer.ChampionExplainer` | 5 |
| Investigation | `fraudlab.agent.investigator.InvestigationAgent` | 6 |
| Dataset generation | `fraudlab.pipeline.build_dataset` | 1 |
| Model metadata | `models/champion_metadata.json` | 4 |

## Common error responses

| Status | Body (`ErrorResponse`) | When |
|---|---|---|
| 422 | `{ "detail": "...validation..." }` | Malformed request body / invalid config |
| 503 | `{ "detail": "champion model unavailable: ..." }` | Required artifact (model/explainer/metadata) missing |
| 500 | `{ "detail": "internal error" }` | Unhandled server error |

---

## 1. `GET /api/health`

- **Purpose:** liveness probe.
- **Method / Route:** `GET /api/health`
- **Request:** none.
- **Response `200` (`HealthOut`):** `{ "status": "ok", "service": "fraudlab-api", "version": "0.1.0" }`
- **Errors:** —
- **Frontend consumer page:** Global (App shell / status indicator).

## 2. `GET /api/model/metadata`

- **Purpose:** champion model metrics & policy for the dashboard.
- **Method / Route:** `GET /api/model/metadata`
- **Request:** none.
- **Response `200` (`ModelMetadataOut`):** champion metadata incl. `metrics` (pr_auc, roc_auc, recall_hard, …), `thresholds`, `confidence_bands`, `training_timestamp`, `dataset_size`, `fraud_rate`.
- **Errors:** `503` (metadata file not found).
- **Frontend consumer page:** Dashboard.

## 3. `POST /api/score`

- **Purpose:** score transactions → fraud probability + decision + confidence.
- **Method / Route:** `POST /api/score`
- **Request (`ScoreRequest`):** `{ "transactions": TransactionIn[] }`
- **Response `200` (`ScoreResponse`):** `{ "predictions": Prediction[] }`
  - `Prediction = { transaction_id, fraud_probability, decision: ALLOW|REVIEW|BLOCK, confidence: LOW|MEDIUM|HIGH }`
- **Errors:** `422` (invalid transactions), `503` (model unavailable).
- **Frontend consumer page:** Transactions / Scoring.

## 4. `POST /api/explain`

- **Purpose:** prediction + SHAP contributors (waterfall data).
- **Method / Route:** `POST /api/explain`
- **Request (`ExplainRequest`):** `{ "transactions": TransactionIn[] }`
- **Response `200` (`ExplainResponse`):** `{ "explanations": Explanation[] }`
  - `Explanation = { transaction_id, fraud_probability, decision, confidence, contributors: Contributor[] }`
  - `Contributor = { feature, impact, direction: increase|decrease }`
- **Errors:** `422`, `503` (explainer unavailable).
- **Frontend consumer page:** Investigation Workspace (SHAP waterfall chart).

## 5. `POST /api/investigate`

- **Purpose:** grounded analyst narratives from pre-computed explanations.
- **Method / Route:** `POST /api/investigate`
- **Request (`InvestigateRequest`):** `{ "explanations": Explanation[], "use_groq": boolean }`
- **Response `200` (`InvestigateResponse`):** `{ "investigations": Investigation[], "mode": "groq"|"fallback" }`
  - `Investigation = { transaction_id, summary, signals_detected[], recommendation, confidence_explanation, next_steps[] }`
- **Errors:** `422` (invalid explanation).
- **Frontend consumer page:** Investigation Workspace.

## 6. `POST /api/transactions/analyze`

- **Purpose:** end-to-end convenience — score + explain + investigate in one call.
- **Method / Route:** `POST /api/transactions/analyze`
- **Request (`AnalyzeRequest`):** `{ "transactions": TransactionIn[], "use_groq": boolean }`
- **Response `200` (`AnalyzeResponse`):** `{ "results": AnalyzedTransaction[], "mode": "groq"|"fallback" }`
  - `AnalyzedTransaction = { transaction_id, prediction, explanation, investigation }`
- **Errors:** `422`, `503` (model/explainer unavailable).
- **Frontend consumer page:** Investigation Workspace (transaction drill-down).

## 7. `POST /api/dataset/generate`

- **Purpose:** generate a synthetic dataset; return summary + preview sample.
- **Method / Route:** `POST /api/dataset/generate`
- **Request (`DatasetConfigIn`):** `{ num_users, transactions_per_user, fraud_rate, business_type, risk_appetite, seed, preview_limit }`
- **Response `200` (`DatasetSummaryOut`):** `{ total_transactions, fraud_count, fraud_percentage, fraud_type_distribution, business_type, num_users, transactions_per_user, random_seed, sample: object[] }`
- **Errors:** `422` (invalid configuration).
- **Frontend consumer page:** Dataset Studio / Simulator.

---

## Schema reference (selected)

### `TransactionIn`
Canonical 25-field simulator schema. `is_fraud` / `fraud_type` optional (ignored at scoring);
extra keys allowed. Required: `transaction_id, user_id, timestamp, amount, currency,
merchant_id, merchant_name, merchant_category, payment_method, channel, device_id,
device_type, city, state, country, is_foreign, account_age_days, avg_user_spend,
median_user_spend, velocity_1h, velocity_24h, merchant_risk_score, device_risk_score`.

### Decision & confidence policy (centralized, Phase 4)
- Decision: `ALLOW` (p < 0.40), `REVIEW` (0.40 ≤ p < 0.80), `BLOCK` (p ≥ 0.80).
- Confidence: `LOW` (p < 0.60), `MEDIUM` (0.60 ≤ p ≤ 0.85), `HIGH` (p > 0.85).

## Security notes
- The investigation agent forwards only transaction *context*, decision outputs, and SHAP
  contributors to Groq — never API keys, model artifacts, or raw datasets.
- `use_groq=false` (or a missing `GROQ_API_KEY`) runs the deterministic offline fallback with
  an identical response shape.
