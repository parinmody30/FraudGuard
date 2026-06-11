"""Tests for the Phase 4 champion-model integration (XGBoost)."""

from __future__ import annotations

import json

import pytest

from fraudlab.config.config import BusinessType, FraudLabConfig, RiskAppetite
from fraudlab.exporters.export import transactions_to_dataframe
from fraudlab.models.champion import (
    ConfidenceBands,
    DecisionThresholds,
    train_champion,
)
from fraudlab.models.persistence import load_bundle, save_bundle
from fraudlab.models.predictor import ChampionPredictor
from fraudlab.pipeline import build_dataset


@pytest.fixture(scope="module")
def sim_df():
    config = FraudLabConfig(
        num_users=150,
        transactions_per_user=40,
        fraud_rate=0.06,
        business_type=BusinessType.BANK,
        risk_appetite=RiskAppetite.MEDIUM,
        random_seed=42,
    )
    transactions, _ = build_dataset(config)
    return transactions_to_dataframe(transactions)


@pytest.fixture(scope="module")
def trained(sim_df):
    return train_champion(sim_df, simulator_config={"seed": 42}, random_state=42)


# ----------------------------------------------------------------- decision policy

def test_decision_thresholds():
    th = DecisionThresholds()
    assert th.decide(0.10) == "ALLOW"
    assert th.decide(0.39) == "ALLOW"
    assert th.decide(0.40) == "REVIEW"
    assert th.decide(0.79) == "REVIEW"
    assert th.decide(0.80) == "BLOCK"
    assert th.decide(0.99) == "BLOCK"


def test_confidence_bands():
    cb = ConfidenceBands()
    assert cb.level(0.50) == "LOW"
    assert cb.level(0.60) == "MEDIUM"
    assert cb.level(0.85) == "MEDIUM"
    assert cb.level(0.86) == "HIGH"


# --------------------------------------------------------------------- persistence

def test_persistence_roundtrip(trained, tmp_path):
    bundle, _ = trained
    mp = tmp_path / "champion.joblib"
    meta_p = tmp_path / "champion_metadata.json"
    paths = save_bundle(bundle, model_path=str(mp), metadata_path=str(meta_p))
    assert mp.exists() and meta_p.exists()

    loaded = load_bundle(model_path=str(mp))
    assert loaded.feature_names == bundle.feature_names
    assert loaded.metadata["model_type"] == "XGBoostClassifier"
    assert loaded.thresholds.block == bundle.thresholds.block

    # Metadata JSON carries the required dashboard fields.
    meta = json.loads(meta_p.read_text())
    for key in ("training_timestamp", "simulator_config", "dataset_size", "fraud_rate"):
        assert key in meta
    for key in ("pr_auc", "roc_auc", "recall", "precision", "f1", "recall_hard"):
        assert key in meta["metrics"]


def test_metadata_metrics_in_range(trained):
    _, metrics = trained
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    # Champion should be a competent detector on this dataset.
    assert metrics["pr_auc"] > 0.5


# ----------------------------------------------------------------------- predictor

def test_predictor_loads_and_predicts(trained, sim_df, tmp_path):
    bundle, _ = trained
    mp = tmp_path / "champion.joblib"
    save_bundle(bundle, model_path=str(mp), metadata_path=str(tmp_path / "m.json"))
    predictor = ChampionPredictor.load(str(mp))

    sample = sim_df.head(20)
    probs = predictor.predict_proba(sample)
    assert len(probs) == 20
    assert ((probs >= 0.0) & (probs <= 1.0)).all()

    labels = predictor.predict(sample)
    assert set(labels.tolist()) <= {0, 1}


def test_batch_prediction_columns(trained, sim_df, tmp_path):
    bundle, _ = trained
    mp = tmp_path / "c.joblib"
    save_bundle(bundle, model_path=str(mp), metadata_path=str(tmp_path / "m.json"))
    predictor = ChampionPredictor.load(str(mp))

    scored = predictor.predict_batch(sim_df.head(50))
    for col in ("fraud_probability", "decision", "confidence"):
        assert col in scored.columns
    assert set(scored["decision"]) <= {"ALLOW", "REVIEW", "BLOCK"}
    assert set(scored["confidence"]) <= {"LOW", "MEDIUM", "HIGH"}
    # Original columns are preserved.
    assert "transaction_id" in scored.columns


def test_decision_confidence_consistency(trained, sim_df, tmp_path):
    bundle, _ = trained
    mp = tmp_path / "c.joblib"
    save_bundle(bundle, model_path=str(mp), metadata_path=str(tmp_path / "m.json"))
    predictor = ChampionPredictor.load(str(mp))

    scored = predictor.predict_batch(sim_df.head(100))
    for _, row in scored.iterrows():
        p = row["fraud_probability"]
        assert row["decision"] == predictor.decision(p)
        assert row["confidence"] == predictor.confidence(p)


def test_frontend_response_contract(trained, sim_df, tmp_path):
    bundle, _ = trained
    mp = tmp_path / "c.joblib"
    save_bundle(bundle, model_path=str(mp), metadata_path=str(tmp_path / "m.json"))
    predictor = ChampionPredictor.load(str(mp))

    records = predictor.predict_response(sim_df.head(5))
    assert len(records) == 5
    for rec in records:
        assert set(rec) == {"transaction_id", "fraud_probability", "decision", "confidence"}
        assert isinstance(rec["fraud_probability"], float)
        assert rec["decision"] in {"ALLOW", "REVIEW", "BLOCK"}
        assert rec["confidence"] in {"LOW", "MEDIUM", "HIGH"}
