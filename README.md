# FraudLab Generator

Enterprise-grade **synthetic financial-transaction dataset generator** for benchmarking
fraud-detection systems, ML research, and synthetic-data studies.

FraudLab produces realistic banking, credit-card, UPI, and e-commerce transaction
datasets with **diverse customer behavior** and **varying fraud difficulty** (easy →
hard). It is deliberately *not* a detector, dashboard, API, or evaluation framework — it
generates labelled datasets only.

> Locale: India-centric (INR, Indian cities/states, realistic UPI ecosystem). `is_foreign`
> marks transactions outside India.

---

## Why this dataset is useful for benchmarking

* **Behavioral realism** — per-user lognormal spend profiles, salary cycles, recurring
  bills, time-of-day/weekday/weekend habits, festival seasonality, account maturation,
  travel episodes, and evolving device pools.
* **Population diversity** — every user has distinct income, spend magnitude/dispersion,
  active-hour archetype, travel and device-churn propensity; populations are not
  homogeneous.
* **Fraud diversity & difficulty** — 14 scenarios across three tiers. Fraud is **not**
  always high-amount / new-device / new-city, and **hard** scenarios are only detectable
  with temporal or sequential context, defeating naive per-row rules.
* **Leakage-free features** — `velocity_1h`, `velocity_24h`, `avg_user_spend`,
  `median_user_spend` are computed *causally* (as-of each transaction), so they are
  features a detector could legitimately use.

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.11+. Dependencies: `numpy`, `pandas`, `pyarrow`, `pyyaml` (+ `pytest`).

---

## Running the web app (backend + frontend)

Prerequisites: Python 3.11+, Node 18+, and (optional) PostgreSQL for dataset history.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
(cd frontend && npm install)
cp .env.example .env          # optionally set GROQ_API_KEY and DATABASE_URL
./start.sh                    # backend on :8000, frontend on :5173
```

**Windows:** double-click `setup-windows.bat` once, then `start-windows.bat` to launch (two terminal
windows open — keep them open while using the app).

Open http://localhost:5173 and sign up (accounts are stored locally in the browser).
API docs are at http://localhost:8000/docs.

* **PostgreSQL** — if `DATABASE_URL` is reachable, generated datasets are saved and listed in
  *Dataset History*; the database is created automatically. Without it, everything else works.
* **Groq** — without `GROQ_API_KEY` the Investigation Agent uses its deterministic offline
  narrative ("Agent mode: FALLBACK"); with a key it writes LLM narratives ("GROQ").
* **CTGAN / PyTorch** are only needed for the GAN training scripts:
  `pip install -r requirements-gan.txt`.

---

## Quick start

Generate from a sample config:

```bash
python scripts/generate_dataset.py --config configs/sample_bank.yaml
```

Generate with inline flags (overrides config / defaults):

```bash
python scripts/generate_dataset.py \
    --business-type upi --num-users 500 --transactions-per-user 90 \
    --fraud-rate 0.02 --risk-appetite low --format parquet --out out/upi.parquet
```

Use it as a library:

```python
from fraudlab import FraudLabConfig, BusinessType, build_dataset
from fraudlab.exporters import export_dataset

config = FraudLabConfig(num_users=500, transactions_per_user=100,
                        fraud_rate=0.02, business_type=BusinessType.BANK,
                        random_seed=42)
transactions, summary = build_dataset(config)
export_dataset(transactions, "out/bank.csv", config.output_format)
print(summary)
```

Generation is **fully deterministic** for a given config + `random_seed`.

---

## Configuration (`FraudLabConfig`)

| Field                   | Meaning                                                        |
|-------------------------|----------------------------------------------------------------|
| `num_users`             | Number of synthetic users                                      |
| `transactions_per_user` | Legitimate transactions generated per user                     |
| `fraud_rate`            | Fraction of the **final** dataset that is fraud (e.g. `0.02`)  |
| `fraud_mix`             | Optional per-scenario weights; if omitted, derived from risk   |
| `business_type`         | `bank` \| `credit_card` \| `upi` \| `ecommerce`                |
| `noise_level`           | `0..1` behavioral jitter to avoid homogeneity                  |
| `risk_appetite`         | `low` \| `medium` \| `high` (shifts default fraud difficulty)  |
| `random_seed`           | Master seed for reproducibility                                |
| `output_format`         | `csv` \| `json` \| `parquet`                                   |
| `start_date`            | Timeline start (ISO date)                                      |
| `currency`              | Default `INR`                                                  |

`business_type` controls the payment-method/channel vocabulary (e.g. UPI → `upi` over
`mobile`/`qr`; credit_card → `credit_card` over `pos`/`online`/`contactless`).

`risk_appetite` default mixes: **low** favors easy fraud, **high** favors subtle/hard
fraud.

---

## Transaction schema

`transaction_id, user_id, timestamp, amount, currency, merchant_id, merchant_name,
merchant_category, payment_method, channel, device_id, device_type, city, state, country,
is_foreign, account_age_days, avg_user_spend, median_user_spend, velocity_1h,
velocity_24h, merchant_risk_score, device_risk_score, is_fraud, fraud_type`

Column order is preserved across all export formats.

---

## Fraud scenarios

| Tier   | Scenarios                                                                       |
|--------|---------------------------------------------------------------------------------|
| Easy   | `account_takeover`, `large_spend`, `geo_anomaly`, `night_time`, `new_device`     |
| Medium | `card_testing`, `velocity`, `merchant`, `beneficiary`                            |
| Hard   | `synthetic_identity`, `money_mule`, `gradual_escalation`, `cross_channel`, `multi_stage` |

Each fraud transaction is a **net addition** on top of pristine baseline activity, so the
realized fraud rate matches the configured rate and labels stay clean.

---

## Output summary

Each run prints (and can write via `--summary-json`):

* total transactions, fraud count, fraud percentage
* fraud-type distribution
* business type, generation time, random seed used

---

## Project layout

```
fraudlab/
  models/        # dataclasses: User, Account, Merchant, Device, Beneficiary, Transaction
  config/        # FraudLabConfig, enums, India reference data
  generators/    # population, merchant, device, beneficiary, baseline transactions
  behaviors/     # spending, temporal, location, velocity behaviors
  fraud/         # scenario registry + engine + easy/medium/hard scenarios
  exporters/     # CSV/JSON/Parquet export with schema integrity
  pipeline.py    # build_dataset(config) -> (transactions, summary)
scripts/         # generate_dataset.py CLI
configs/         # sample YAML configs per business type
tests/           # pytest suite
```

---

## Testing

```bash
pytest -q
```

Tests cover deterministic generation, schema correctness, fraud rate/mix correctness,
export integrity, behavioral diversity, seed reproducibility, and id uniqueness.

---

## Performance

Optimized for realism over speed, but generating ~100k transactions completes in a few
minutes on a standard laptop (well within the 10-minute target). Example:

```bash
python scripts/generate_dataset.py --num-users 1000 --transactions-per-user 100 \
    --fraud-rate 0.02 --format parquet --out out/large.parquet
```

---

## Evaluation framework (Phase 1.1)

`fraudlab.evaluation` is a read-only evaluator (it never mutates the generator). It reports
descriptive metrics, a user-grouped detector benchmark with per-difficulty-tier recall,
label-leakage detection, a sequential-complexity analysis, and a CTGAN-readiness verdict.

```bash
pip install -r requirements.txt          # adds scikit-learn / scipy
python scripts/evaluate_dataset.py --input out/bank_seed42.csv
python scripts/evaluate_dataset.py --generate --num-users 800 --fraud-rate 0.02
```

## CTGAN realism augmenter (Phase 2)

`fraudlab.gan` trains [CTGAN](https://github.com/sdv-dev/CTGAN) as a **per-row realism
augmenter** for legitimate + easy + medium transactions. It never replaces the simulator
and is deliberately **not** trained on the five hard (sequential) scenarios — those remain
simulator-generated and are spliced back in by the merger to form a hybrid dataset. The
generator code (`generators/`, `fraud/`, `behaviors/`, `pipeline.py`), the schema, and the
export formats are untouched.

```bash
# Train, sample, evaluate, and build a hybrid dataset (needs the `gan` extra: ctgan, torch)
python scripts/train_ctgan.py    --generate --num-users 250 --transactions-per-user 50 \
                                 --fraud-rate 0.08 --epochs 120 --seed 42
python scripts/sample_ctgan.py   --n 2000 --fraud-rate 0.05 --out out/ctgan_mixed.csv
python scripts/evaluate_ctgan.py --generate --num-users 250 --transactions-per-user 50 \
                                 --fraud-rate 0.08 --sim-seed 42
```

Evaluation writes `reports/ctgan_evaluation.{json,md}` and `reports/ctgan_similarity.csv`,
and prints a **READY FOR HYBRID GENERATION** / **CTGAN NEEDS TUNING** recommendation.

## Groq Investigation Agent configuration (Phase 6 / 7.2)

The Investigation Agent (`fraudlab.agent`) turns champion + SHAP evidence into analyst
narratives. It uses **Groq** (`llama-3.3-70b-versatile`) when a key is available and a
**deterministic offline fallback** otherwise — both return the identical response contract.

**Enable Groq (permanent):** the FastAPI backend auto-loads a project-root `.env` on startup
(`backend/app.py` calls `load_dotenv(...)`), so the key is read from the environment at
request time via `os.getenv("GROQ_API_KEY")` — never hardcoded.

```bash
cp .env.example .env          # then edit .env:
# GROQ_API_KEY=gsk_your_key_here
uvicorn backend.app:app --reload
```

- **Key present & valid** → live Groq narratives (`mode: "groq"`).
- **Key missing** → deterministic fallback (`mode: "fallback"`); backend still starts.
- **Key invalid / Groq request fails** → caught automatically, falls back gracefully; no
  stack traces leak to users; the frontend response shape is unchanged.

`recommendation` and `next_steps` are always derived from the decision policy (never the
LLM). Security: `.env` is gitignored; keys are never logged, written to reports, or exposed
through any endpoint. Only transaction context, decision outputs, and SHAP contributors are
sent to Groq. For a fully offline demo, simply omit `GROQ_API_KEY`.

## Scope

Phase 1: population simulation, baseline transactions, fraud engine, configuration, export
pipeline, tests, docs. Phase 1.1: the read-only evaluation framework above. Phase 2: the
optional CTGAN realism augmenter + hybrid merger above. Phases 3–7: detector benchmarking,
champion (XGBoost) operationalization, SHAP explainability, the Groq Investigation Agent,
and the FastAPI + React integration layer. Out of scope (by design): databases, auth,
sequence-aware GANs.
